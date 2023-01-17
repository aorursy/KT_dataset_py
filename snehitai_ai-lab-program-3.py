# Bredth first search in search of target - Using Brute Force Algorithms
def bfs(src, target):
    visited_states=[]
    visited_states.append(src)
    arr = [src]
    c = 0
    while arr:
        c += 1                                           # Calculate Number of Iterations                                   # Print current state to check
        if arr[0] == target:                             # break if target found
            return True
        arr += possible_moves(arr[0],visited_states)                    # else Add all possible moves to arr
        visited_states.append(arr[0])  
        arr.pop(0)                                       # remove checked move from arr                                  
    return False
# Find Possible Moves
def possible_moves(state,visited_states): 
    ind = state.index(-1)
    d=[]
    if ind+3 in range(9):
        d.append('d')
    if ind-3 in range(9):
        d.append('u')
    if ind not in [0,3,6]:
        d.append('l')
    if ind not in [2,5,8]:
        d.append('r')
    pos_moves=[]
    for move in d:
        pos_moves.append(gen(state,move,ind))
    return [move for move in pos_moves if move not in visited_states]


# Generate move for given direction
def gen(state, direction, b):
    temp=state.copy()
    if direction=='d':
        a = temp[b+3]
        temp[b+3]=temp[b]
        temp[b]=a
    elif direction=='u':
        a = temp[b-3]
        temp[b-3]=temp[b]
        temp[b]=a
    elif direction=='l':
        a = temp[b-1]
        temp[b-1]=temp[b]
        temp[b]=a
    elif direction=='r':
        a = temp[b+1]
        temp[b+1]=temp[b]
        temp[b]=a
    return temp
#Test 1
src = [1,2,3,-1,4,5,6,7,8]
target = [1,2,3,4,5,-1,6,7,8]         
       


bfs(src, target) 
# Test 2
src = [1,2,3,-1,4,5,6,7,8] 
target=[1,2,3,6,4,5,-1,7,8]



bfs(src, target)