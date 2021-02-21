# 205956634 Yuval Saadaty
import math


class State:
    """
    The name of the cass is state but it is actually represents a node.
    Each node has the files:
    row -Index of the state row
    column - Index of the state column
    val - The value of a state which is the cost to get to that state
    depth - The depth of the node in the graph. The depth of each node
     will be the depth of his parent plus 1.
    path_cost - represents g function that evaluate the cost from start
     node to that current node.
     h - Is a heuristic function
     f = h funcion + path_cost
     node_path = path from arart node to current node
    """
    def __init__(self, row, column, val):
        self.row = row
        self.column = column
        self.val = val
        self.depth = -1
        self.path_cost = 0
        self.h = 0
        self.f = 0
        self.node_path = list()


algo_type = None # IDS, UCS, ASTAR, IDASTAR
start_state = None # start state
end_state = None # end state
board_size =None # board size
path_lenght = 0 # final path length
board = [] # The domain is a board. The board composed of list of lists


def loadDataToBoard():
    """
    The board is list of lists and in each cell will be object type of State.
    All states will be created in that function, include start state and end state.
    """
    global board, board_size, end_state, start_state, algo_type
    board = [] # lst of lists
    file_list = list() # single list for each row
    file_input = open("input.txt", "r") # open the problem data
    for line in file_input:  # create list from input.txt
        file_list.append(line)
    algo_type = file_list[0]  # algorithm type
    start_point = list(map(int, file_list[1].split(",")))  # start state
    end_point = list(map(int, file_list[2].split(",")))  # end state
    board_size = int(file_list[3])  # board size
    for row in range(4, board_size + 4):
        column = -1
        row_list = [] # new list for new line in board matrix
        for val in list(map(int, file_list[row].split(","))):
            column += 1
            new_state = State(row-4, column, val) # new state cell
            new_state.h = chebyshev(row-4, column, end_point[0], end_point[1])
            row_list.append(new_state)
        board.append(row_list) # add new row to matrix
    start_state = board[start_point[0]][start_point[1]]
    start_state.depth = 0
    end_state = board[end_point[0]][end_point[1]]


def getChildren(curr_state):
    """
    The input is a single node. The function will expand all the
     nodes children and will return a list of them.
     The children have different priority, the right most child has the highest priority.
     Before insert a child to children list we will check if the value of the child is different from -1
     and if the child is diagonally to his parent so there is no -1 state from his 2 sides.
     We will use this function only in IDS algorithm because in IDS we will insert the same child
     multiple times so we need to create new object to every child and not using the same object with
     diffrent values.
     :param curr_state: single state
    :return: list of valid children
    """
    global board_size
    children = []
    i = curr_state.row # row index in this state
    j = curr_state.column  # column index in this state
    # insert children to children list when the right state is preferred.
    # Right
    if i >= 0 and i < board_size and j >= 0 and j+1 < board_size:
        if board[i][j+1].val != -1:
            children.append(State(board[i][j+1].row, board[i][j+1].column, board[i][j+1].val))
    # RD
    if i >= 0 and i+1 < board_size and j >= 0 and j+1 < board_size:
        if board[i+1][j+1].val != -1:
            if hasBadRightSon(curr_state) is False and hasBadDownSon(curr_state) is False:
                children.append(State(board[i+1][j+1].row, board[i+1][j+1].column, board[i+1][j+1].val))
    # D
    if i >= 0 and i+1 < board_size and j >= 0 and j < board_size:
        if board[i+1][j].val != -1:
            children.append(State(board[i+1][j].row, board[i+1][j].column, board[i+1][j].val))
    # DL
    if i >=0 and i+1 < board_size and j >= 1 and j-1 < board_size:
        if board[i+1][j-1].val != -1:
            if hasBadDownSon(curr_state) is False and hasBadLeftSon(curr_state) is False:
                children.append(State(board[i+1][j-1].row, board[i+1][j-1].column, board[i+1][j-1].val))
    # L
    if i >= 0 and i < board_size and j >= 1 and j-1 < board_size:
        if board[i][j-1].val != -1:
            children.append(State(board[i][j-1].row, board[i][j-1].column, board[i][j-1].val))
    # LU
    if i >= 1 and i -1 < board_size and j >= 1 and j-1 < board_size:
        if board[i-1][j-1].val != -1:
            if hasBadLeftSon(curr_state) is False and hasBadUpSon(curr_state) is False:
                children.append(State(board[i-1][j-1].row, board[i-1][j-1].column, board[i-1][j-1].val))
    # U
    if i >= 1 and i -1< board_size and j >= 0 and j < board_size:
        if board[i-1][j].val != -1:
            children.append(State(board[i-1][j].row, board[i-1][j].column, board[i-1][j].val))
    # UR
    if i >= 1 and i-1 < board_size and j >= 0 and j+1 < board_size:
        if board[i-1][j+1].val != -1:
            if hasBadRightSon(curr_state) is False and hasBadUpSon(curr_state) is False:
                children.append(State(board[i-1][j+1].row, board[i-1][j+1].column, board[i-1][j+1].val))
    return children


def getChildrenUCS(curr_state):
    """
    The input is a single node. The function will expand all the
     nodes children and will return a list of them.
     The children have different priority, the right most child has the highest priority.
     Before insert a child to children list we will check if the value of the child is different from -1
     and if the child is diagonally to his parent so there is no -1 state from his 2 sides.
     We will use this function in UCS, ASTAR and IDASTAR algorithms because in those algorithms we nedd to save
     the files of each node, so each node will be created once in creatBoard function and will be update
     during running.
     :param curr_state: single state
    :return: list of valid children
    """
    global board_size
    children = []
    i = curr_state.row
    j = curr_state.column
    # Right
    if i >= 0 and i < board_size and j >= 0 and j+1 < board_size:
        if board[i][j+1].val != -1:
            children.append(board[i][j+1])
    # RD
    if i >= 0 and i+1 < board_size and j >= 0 and j+1 < board_size:
        if board[i+1][j+1].val != -1:
            if hasBadRightSon(curr_state) is False and hasBadDownSon(curr_state) is False:
                children.append(board[i+1][j+1])
    # D
    if i >= 0 and i+1 < board_size and j >= 0 and j < board_size:
        if board[i+1][j].val != -1:
            children.append(board[i+1][j])
    # DL
    if i >=0 and i+1 < board_size and j >= 1 and j-1 < board_size:
        if board[i+1][j-1].val != -1:
            if hasBadDownSon(curr_state) is False and hasBadLeftSon(curr_state) is False:
                children.append(board[i+1][j-1])
    # L
    if i >= 0 and i < board_size and j >= 1 and j-1 < board_size:
        if board[i][j-1].val != -1:
            children.append(board[i][j-1])
    # LU
    if i >= 1 and i -1 < board_size and j >= 1 and j-1 < board_size:
        if board[i-1][j-1].val != -1:
            if hasBadLeftSon(curr_state) is False and hasBadUpSon(curr_state) is False:
                children.append(board[i-1][j-1])
    # U
    if i >= 1 and i -1< board_size and j >= 0 and j < board_size:
        if board[i-1][j].val != -1:
            children.append(board[i-1][j])
    # UR
    if i >= 1 and i-1 < board_size and j >= 0 and j+1 < board_size:
        if board[i-1][j+1].val != -1:
            if hasBadRightSon(curr_state) is False and hasBadUpSon(curr_state) is False:
                children.append(board[i-1][j+1])
    return children


def hasBadRightSon(curr_state):
    """
    Check if state has right son with value -1.
    :param curr_state: single state
    :return: True if there is -1 states right to curr_state and false otherwise.
    """
    global board_size
    i = curr_state.row
    j = curr_state.column
    # has bad son Right
    if i >= 0 and i < board_size and j >= 0 and j + 1 < board_size:
        if board[i][j + 1].val == -1:
            return True
    return False


def hasBadDownSon(curr_state):
    """
    Check if state has down son with value -1.
    :param curr_state: single state
    :return: True if there is -1 states down to curr_state and false otherwise.
    """
    global board_size
    i = curr_state.row
    j = curr_state.column
    # Down
    if i >= 0 and i + 1 < board_size and j >= 0 and j < board_size:
        if board[i + 1][j].val == -1:
            return True
    return False


def hasBadLeftSon(curr_state):
    """
    Check if state has left son with value -1.
    :param curr_state: single state
    :return: True if there is -1 states left to curr_state and false otherwise.
    """
    global board_size
    i = curr_state.row
    j = curr_state.column
    # Left
    if i >= 0 and i < board_size and j >= 1 and j - 1 < board_size:
        if board[i][j - 1].val == -1:
            return True
    return False


def hasBadUpSon(curr_state):
    """
    Check if state has up son with value -1.
    :param curr_state: single state
    :return: True if there is -1 states up to curr_state and false otherwise.
    """
    global board_size
    i = curr_state.row
    j = curr_state.column
    # Up
    if i >= 1 and i - 1 < board_size and j >= 0 and j < board_size:
        if board[i - 1][j].val == -1:
            return True
    return False


def printPath(path):
    """
    Pass on path from start to end and evaluate the next step by the given indexes.
    :param path: path that the first element is start state and last element is end state.
    :return: string that represents the steps
    """
    path_string = ""
    for i in range(len(path)-1):
        # down
        if path[i].row == path[i+1].row - 1 and path[i].column == path[i+1].column:
            path_string += "D-"
        # up
        elif path[i].row == path[i+1].row + 1 and path[i].column == path[i+1].column:
            path_string += "U-"
        # Right
        elif path[i].row == path[i + 1].row and path[i].column == path[i + 1].column -1:
            path_string += "R-"
        # Left
        elif path[i].row == path[i + 1].row and path[i].column == path[i + 1].column + 1:
            path_string += "L-"
        # Right Down
        elif path[i].row == path[i + 1].row -1 and path[i].column == path[i + 1].column - 1:
            path_string += "RD-"
        # Left Down
        elif path[i].row == path[i + 1].row - 1 and path[i].column == path[i + 1].column + 1:
            path_string += "LD-"
        # Left Up
        elif path[i].row == path[i + 1].row + 1 and path[i].column == path[i + 1].column + 1:
            path_string += "LU-"
        # Right Up
        elif path[i].row == path[i + 1].row + 1 and path[i].column == path[i + 1].column - 1:
            path_string += "RU-"
    # delete last "-", no need
    path_string = path_string[:-1]
    return str(path_string)


def evaluatePath(path):
    """
    Pass on all path states and sum up each state value.
    This function will by used for the output.
    :param path: path that the first element is start state and last element is end state.
    :return: sum of the values of every state, without start state.
    """
    sum = 0
    for i in range(1, len(path)):
        sum += path[i].val
    return sum


def getFinalPath(path):
    """
    Pass all elements in the path and when there is state tha the depth of her
    is smaller then he one that before her, it is mean that we need to go back to point
    where the current state has the same depth like one before. By that we create the real path
    from start to end without redundant nodes.
    :param path: path that the first element is start state and last element is end state.
    :return: the real path that solved the problem.
    """
    state = 0
    while state < len(path)-1:
        if len(path) > 1:
            if path[state].depth != path[state+1].depth - 1:
                temp_depth = path[state+1].depth
                while path[state].depth != temp_depth:
                    path.remove(path[state])
                    state -= 1
                path.remove(path[state])
                state -= 1
        state += 1
    return path


def getMinVal(frontier):
    """
    Pass on all nodes in frontier list, find the smallest one and delete him from list.
    :param frontier: open list of nodes
    :return: the node with the smallest value and new frontier list without that node.
    """
    temp_frontier = frontier
    save_index = 0
    if frontier:
        save_min = frontier[0]
        min = save_min.path_cost
        for i in range(len(frontier)):
            if frontier[i].path_cost <= min:
                min = frontier[i].path_cost
                save_min = frontier[i]
                save_index = i
        del temp_frontier[save_index]
        return temp_frontier, save_min
    return None


def depthLimitedSearch(limit):
    """
    Implement depth limited search algorithm.
    :param limit: limit from 0 to 20
    :return: path from start to end to solve the problem, otherwise None
    """
    global start_state, end_state, path_lenght
    nodes_path = []
    frontier = [start_state]
    while len(frontier) != 0:
        curr_node = frontier.pop()
        # all nodes from open list
        nodes_path.append(curr_node)
        if curr_node.row == end_state.row and curr_node.column == end_state.column:
            # found the end state
            path_lenght += len(nodes_path)
            return nodes_path
        if curr_node.depth < limit:
            children_states = getChildren(curr_node)
            children_states.reverse() # reverse because using stack
            for state in children_states:
                state.depth = curr_node.depth + 1
                frontier.append(state)
    # sum up all nodes that removed from open list
    path_lenght += len(nodes_path)
    return None


def iterativeDepthSearch():
    """
    Increase the limit of depthLimitedSearch by 1 until the
     end will be reached or until 20 interation.
    :return: solution- a path from start to end to solve the problem
    """
    for depth in range(0, 20):
        loadDataToBoard()
        result = depthLimitedSearch(depth)
        if result is not None:
            return getFinalPath(result)
    return None


def BestFirstGraphSearch():
    """
    Implement BestFirstGraphSearch algorithem
    :return: solution- a path from start to end to solve the problem
    """
    global path_lenght, start_state, end_state
    child_father_dic = {}
    frontier = [] # initializing list - priority Queue
    start_state.depth = 0
    frontier.append(start_state)
    closed_list = list()
    counter_open_list = 0 # count the amount of nodes that pop out from open list
    while len(frontier) > 0:
        # frontier - the same list just without the smallest value of node
        # node - the smallest node in frontier
        frontier, node = getMinVal(frontier)
        counter_open_list += 1
        if node.row == end_state.row and node.column == end_state.column:
            # found the end state
            path_lenght = counter_open_list -1
            node.node_path.append(end_state)
            return node.node_path
        closed_list.append(node)
        children = getChildrenUCS(node)
        for child in children:
            # find child index
            index = getIndex(frontier, child)
            new_cost = child.val + node.path_cost
            if child not in frontier and child not in closed_list:
                child.depth = node.depth + 1
                child.path_cost += new_cost
                frontier.append(child)
                # add parnt to child path
                for temp in node.node_path:
                    child.node_path.append(temp)
                child.node_path.append(node)
            elif child in frontier and new_cost < frontier[index].path_cost:
                child.path_cost = new_cost
                child.node_path.append(node)
    return None


def uniformCostSearch():
    """
    :return: call BestFirstGraphSearch function
    """
    return BestFirstGraphSearch()


def manhattan(state_x, state_y, goal_x, goal_y):
    """
    :return: calculate distance between two points by manhattan distance.
    """
    return abs(goal_x - state_x) + abs(goal_y - state_y)


def octalDistance(state_x, state_y, goal_x, goal_y):
    """
    :return: calculate distance between two points by octal distance.
    """
    dy = state_y - goal_y
    dy = dy * dy
    dx = state_x - goal_x
    dx = dx * dx
    return math.sqrt(dy + dx)


def chebyshev(state_x, state_y, goal_x, goal_y):
    """
    :return: calculate distance between two points by chebyshev distance.
    """
    dy = abs(state_y - goal_y)
    dx = abs(state_x - goal_x)
    return (dy + dx) + (-1 * min(dx, dy))


def getIndex(frontier, child):
    """
    Pass of frontier nodes to find the child index in the list.
    :param frontier: list of nodes
    :param child: a single node
    :return: the index of child in frontier
    """
    i = -1
    index = -1
    for s in frontier:
        i += 1
        if s.row == child.row and s.column == child.column:
            index = i
    return index


def AstarSearch():
    """
    Implement A* search algorithm
    :return: solution- a path from start to end to solve the problem
    """
    global path_lenght, start_state, end_state
    frontier = [] # initializing list - priority Queue
    frontier.append(start_state)
    closed_list = set()
    counter_open_list = 0 # count the amount of nodes that pop out from open list
    while len(frontier) > 0 :
        # get node the minimum value and frontier is a list without that node
        frontier, node = getMinVal(frontier)
        counter_open_list += 1
        if node.row == end_state.row and node.column == end_state.column:
            # path length without first and last nodes
            path_lenght = counter_open_list - 1
            node.node_path.append(end_state)
            return node.node_path
        closed_list.add(node)
        children = getChildrenUCS(node)
        for child in children:
            # find child index
            index = getIndex(frontier, child)
            new_cost = child.val + node.path_cost
            new_f = child.h + new_cost
            if child not in closed_list and child not in frontier:
                child.depth = node.depth + 1
                child.path_cost += new_cost
                child.f = new_f
                frontier.append(child)
                for temp in node.node_path:
                    child.node_path.append(temp)
                child.node_path.append(node)
            elif child in frontier and new_f < frontier[index].f:
                    # update child files
                    child.path_cost = new_cost
                    child.f = new_f
                    child.node_path.append(node)
    return None


def DFSf(state, path, f_limit, new_limit, depth):
    """
    Implement DFS algorithm.
    :param state: current state
    :param path: will be the final path from start to end
    :param f_limit: current limit
    :param new_limit: new limit, will be initializ with infi
    :param depth: current node depth
    :return: solution- a path from start to end to solve the problem, None otherwise
    """
    global path_lenght, start_state, end_state
    new_f = state.path_cost + state.h
    if new_f > f_limit[0]:
        new_limit[0] = min(new_limit[0], new_f)
        return None
    if end_state.row == state.row and end_state.column == state.column:
        return path
    for child in getChildrenUCS(state):
        if child not in path:
            path.append(child)
            child.path_cost = child.val + state.path_cost
            child.f = child.path_cost + child.h
            child.depth = state.depth + 1
            depth[0] = child.depth
            path_lenght += 1
            solution = DFSf(child, path, f_limit, new_limit, depth)
            if solution:
                return solution
            path.pop()
    return None


def IDAstarSearch():
    """
    Implement ID A* search algorithm. The number of iteration will
     be dependent on the last node depth. The max depth that the algorithm can reach is 20.
    :return: solution - a path from start to end to solve the problem, None otherwise
    """
    depth = [0]
    f_limit = [start_state.h]
    new_limit = [start_state.h]
    while depth[0] < 21:
        loadDataToBoard()
        f_limit[0] = new_limit[0]
        new_limit[0] = float("inf")
        path = [start_state]
        solution = DFSf(start_state, path, f_limit, new_limit, depth)
        if solution:
            return path
    return None


if __name__ == '__main__':
    output_line = ""
    path = ""
    # convert data of a problem to board matrix
    loadDataToBoard()
    if algo_type == "IDS\n":
        path_lenght -= 1
        path = iterativeDepthSearch()
    elif algo_type == "UCS\n":
        path = uniformCostSearch()
    elif algo_type == "ASTAR\n":
        path = AstarSearch()
    elif algo_type == "IDASTAR\n":
        path = IDAstarSearch()
    if path is None:
        output_line = "no path"
    else:
        path_string = printPath(path)
        path_cost = evaluatePath(path)
        output_line = str(path_string) + " " + str(path_cost) + " " + str(path_lenght)
    output_file = open("output.txt", "w+")
    output_file.write(output_line)
