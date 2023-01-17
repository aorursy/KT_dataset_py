def G_n(state):
    # Note: You can skip this step if you feel you have a better way of getting G(n)
    pass
def H_n(state, target):
    pass
def F_n():# Fill inputs as necessary
    pass
    
def astar(state, target):# Add inputs if more are required
    pass
def possible_moves(state):# Add inputs if more are required
    pass

def gen(state, direction, blank_index):
    pass
#Test 1
src = [1,2,3,-1,4,5,6,7,8]
target = [1,2,3,4,5,-1,6,7,8]         
       


astar(src, target) 
# Test 2
src = [1,2,3,-1,4,5,6,7,8] 
target=[1,2,3,6,4,5,-1,7,8]



astar(src, target)
# Test 3
src = [1,2,3,7,4,5,6,-1,8] 
target=[1,2,3,6,4,5,-1,7,8]



astar(src, target)