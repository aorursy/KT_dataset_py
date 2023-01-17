cost = 1

deltas = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1]] # go right

delta_name = ['^', '<', 'v', '>']
def index_path_with_less_cost(paths):
    path_costs = [path[0] for path in paths]
    min_cost = min(path_costs)
    return path_costs.index(min_cost)        

def is_goal(position, goal):
    return position[1] == goal[0] and position[2] == goal[1]

def move(position, delta):
    return [position[1] + delta[0], position[2] + delta[1]]

def is_in_grid(position, grid):
    in_grid = position[0] >= 0 and position[0] < len(grid) and position[1] >= 0 and position[1] < len(grid[0])
    if (in_grid):
        return grid[position[0]][position[1]] == 0
    else:
        return 0

def is_in_world(position, world):
    return world[position[0]][position[1]] == 0
    
def create_open_positions(grid, world, position):
    open_positions = []
    for delta in deltas:
        candidate_position = move(position, delta)
        if (is_in_grid(candidate_position, grid) and is_in_world(candidate_position, world)):
            world[candidate_position[0]][candidate_position[1]] = 1
            open_positions.append([position[0] + cost, candidate_position[0], candidate_position[1]])
    return open_positions
    
def search(grid,init,goal,cost):
    open_positions = [[0, init[0], init[1]]]
    world = [[0 for y in range(len(grid[0]))] for x in range(len(grid))]
    while (len(open_positions) > 0):
        next_position_index = index_path_with_less_cost(open_positions)
        next_position = open_positions[next_position_index]
        del open_positions[next_position_index]
        
        if (is_goal(next_position, goal)):
            return next_position

        open_positions.extend(create_open_positions(grid, world, next_position))
    
    return "fail"
# ----------
# User Instructions:
# 
# Define a function, search() that returns a list
# in the form of [optimal path length, row, col]. For
# the grid shown below, your function should output
# [11, 4, 5].
#
# If there is no valid path from the start point
# to the goal, your function should return the string
# 'fail'
# ----------

# Grid format:
#   0 = Navigable space
#   1 = Occupied space

grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0]]
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]

print(search(grid,init,goal,cost))