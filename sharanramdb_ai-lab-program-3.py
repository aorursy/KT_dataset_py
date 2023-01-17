# Bredth first search in search of target - Using Brute Force Algorithms

def bfs(src,target):
    queue = []
    queue.append(src)
    
    exp = []
    
    while len(queue) > 0:
        source = queue.pop(0)
        exp.append(source)
        
        print(source)
        
        if source==target:
            print("success")
            return
        
        poss_moves_to_do = []
        poss_moves_to_do = possible_moves(source,exp)
        
        for move in poss_moves_to_do:
            
            if move not in exp and move not in queue:
                queue.append(move)
        
    
    
def possible_moves(state,visited_states): 
    # Find index of empty spot and assign it to b
    b = state.index(-1)
    
    #'d' for down, 'u' for up, 'r' for right, 'l' for left - directions array
    d = []
    #Add all possible direction into directions array - Hint using if statements
    
    if b not in [0,1,2]: 
        d.append('u')
    if b not in [6,7,8]: 
        d.append('d')
    if b not in [0,3,6]: 
        d.append('l')
    if b not in [2,5,8]: 
        d.append('r')
        

    # If direction is possible then add state to move
    pos_moves_it_can = []
    
    # for all possible directions find the state if that move is played
    ### Jump to gen function to generate all possible moves in the given directions
    
    for i in d:
        pos_moves_it_can.append(gen(state,i,b))
        
    return [move_it_can for move_it_can in pos_moves_it_can if move_it_can not in visited_states]
# Generate move for given direction
def gen(state, m, b):
    temp = state.copy()                              
    
    # if move is to slide empty spot to the left and so on
    
    
    if m=='d':
        temp[b+3],temp[b] = temp[b],temp[b+3]
    
    if m=='u':
        temp[b-3],temp[b] = temp[b],temp[b-3]
    
    if m=='l':
        temp[b-1],temp[b] = temp[b],temp[b-1]
    
    if m=='r':
        temp[b+1],temp[b] = temp[b],temp[b+1]
        
    
    # return new state with tested move to later check if "src == target"
    return temp
#Test 1
src = [1,2,3,-1,4,5,6,7,8]
target = [1,2,3,4,5,-1,6,7,8]         
       


bfs(src, target) 
# Test 2
src = [1,2,3,-1,4,5,6,7,8] 
target=[1,2,3,6,4,5,-1,7,8]



bfs(src, target)