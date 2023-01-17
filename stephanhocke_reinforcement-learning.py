import numpy as np
from PIL import Image, ImageDraw
import math
from numpy.random import MT19937f
from numpy.random import RandomState, SeedSequence 
# create some random tsp points scattered within 190x190
np.random.seed(123)

n = 15
xc = np.random.randint(low = 0, high = 190, size = n)
yc = np.random.randint(low = 0, high = 190, size = n)

# merge coordinates to points
cc = np.zeros( (n,2) )
cc[:,0] = xc
cc[:,1] = yc

cities = [i for i in range(0,n)] # auxillary variable
dists = [ [math.sqrt(pow(p1[0]-p2[0],2)+pow(p1[1]-p2[1],2)) for i,p1 in enumerate(cc)] for j,p2 in enumerate(cc)] # distances
# auxillary function to calculate the average edge length
def compute_avg_edge_length(dists):
    total = 0.0
    for i in range(0,n-1):
        for j in range(i+1,n):
            total += dists[i][j]
    return total/(n*(n-1)/2)
# computes the tour lenght of a path list object
def compute_tour_length(path):
    length = 0.0
    for i in range(0,n):
        length = length + dists[int(path[i])][int(path[i+1])]
    return length
# draw the next action for an agent
def draw_next_city(path):
    curIdx = path[len(path)-1]
    possible = [x for x in cities if x not in path]

    if np.random.uniform(0,1) < 0.9:
        values = [pow(Q[curIdx,j],rho)*pow(HE[curIdx][j],beta) for j in possible]
        nextIdx = possible[values.index(max(values))]
    else:
        nextIdx = possible[np.random.randint(0,len(possible))]
    return nextIdx
# learning parameter
gamma = 0.3
rho = 1
beta = 2
alpha = 0.1

# Q-table and heuristic value table
Q = np.matrix(np.ones([n,n]))
Q = Q/(compute_avg_edge_length(dists)*n)
HE = [ [0 if math.sqrt(pow(p1[0]-p2[0],2)+pow(p1[1]-p2[1],2))==0 else 1/math.sqrt(pow(p1[0]-p2[0],2)+pow(p1[1]-p2[1],2)) for i,p1 in enumerate(cc)] for j,p2 in enumerate(cc)]
LgBest = compute_tour_length([i for i in range(0,n)]+[0])
LgPath = []

print("Best solution so far {}".format(LgBest))


for it in range(0,1000):
    LiBest = 1000000
    LiPath = []
    paths = []
    for agent in range(0,n):
        p = [agent]
        for j in range(0,n-1):
            lastIdx = p[len(p)-1]
            nextIdx = draw_next_city(p)
            Q[lastIdx,nextIdx] = (1-alpha)*Q[lastIdx,nextIdx]+alpha*gamma*max([Q[nextIdx,x] for x in p])
            p = p + [nextIdx]
        p = p + [agent]

        print(p)
        paths.append(p)

        fitness = compute_tour_length(p)
        print("current solution {}".format(fitness))

        if fitness < LgBest:
            LgBest = fitness
            LgPath = p
        if fitness < LiBest:
            LiBest = fitness
            LiPath = p

    for j in range(0,n):
        r = int(LiPath[j])
        s = int(LiPath[j+1])
        Q[r,s] = (1-alpha)*Q[lastIdx,nextIdx]+alpha*gamma*max([Q[nextIdx,x] for x in p])*10/LiBest


print("Best fitness {}".format(LgBest))
edges = [(LgPath[idx],LgPath[idx+1]) for idx in range(0,n)]
print(edges)
# init canvas
im = Image.new('RGB', (200,200) )
draw = ImageDraw.Draw(im)


# draw best tour
for i in range(0,n):
    line = [cc[LgPath[i],:],cc[LgPath[i+1],:]]
    draw.line( [ (line[0][0],line[0][1]), (line[1][0],line[1][1]) ], fill= "white", width=1)

im.show()