# Bredth first search in search of target - Using Brute Force Algorithms

def print_grid(src):
    state = src.copy()
    state[state.index(-1)] = ' '
    print(
        f"""
{state[0]} {state[1]} {state[2]}
{state[3]} {state[4]} {state[5]}
{state[6]} {state[7]} {state[8]}
        """
    )


def bfs(src,target):
    frontier = [src]
    visited_states = set()
    while len(frontier):
        state = frontier.pop(0)
        print_grid(state)
        visited_states.add(tuple(state))
        if state == target:
            print(f"Success")
            return
        for move in possible_moves(state, visited_states):
            if move not in frontier and tuple(move) not in visited_states:
                frontier.append(move)
    print("Fail")
        
    
    
# Find Possible Moves
def possible_moves(state, visited_states): 
    b = state.index(-1)  
    d = []
    if 9 > b - 3 >= 0: 
        d += 'u'
    if 9 > b + 3 >= 0:
        d += 'd'
    if b not in [2,5,8]: 
        d += 'r'
    if b not in [0,3,6]: 
        d += 'l'
    pos_moves = []
    for move in d:
        pos_moves.append(gen(state,move,b))
    return [move for move in pos_moves if tuple(move) not in visited_states]


# Generate move for given direction
def gen(state, direction, blank_spot):
    temp = state.copy()                              
    if direction == 'u':
        temp[blank_spot-3], temp[blank_spot] = temp[blank_spot], temp[blank_spot-3]
    if direction == 'd':
        temp[blank_spot+3], temp[blank_spot] = temp[blank_spot], temp[blank_spot+3]
    if direction == 'r':
        temp[blank_spot+1], temp[blank_spot] = temp[blank_spot], temp[blank_spot+1]
    if direction == 'l':
        temp[blank_spot-1], temp[blank_spot] = temp[blank_spot], temp[blank_spot-1]
    return temp
#Test 1
src = [1,2,3,-1,4,5,6,7,8]
target = [1,2,3,4,5,-1,6,7,8]         
       

bfs(src, target) 
# Test 2
src = [1,2,3,-1,4,5,6,7,8] 
target=[1,2,3,6,4,5,-1,7,8]



bfs(src, target)