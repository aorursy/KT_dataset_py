import random

def randomwalk(n): 
    x,y=0,0
    for i in range(n):
        (dx,dy) = random.choice([(0,1),(0,-1),(-1,0),(1,0)])
        x+=dx
        y+=dy
    return x,y
    
number_of_walks = 20000

for walk_length in range (1, 50):
    total_distance=0
    for i in range(number_of_walks):
        (x,y)=randomwalk(walk_length)
        distance = abs(x)+abs(y)
        total_distance+=distance
    average_walk=total_distance/number_of_walks
    print("walk size =",walk_length,"average distance (as block) from home = ",average_walk)
    


