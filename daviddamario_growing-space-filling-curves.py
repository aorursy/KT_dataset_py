import numpy as np



import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import collections  as mc



import seaborn as sns



from tqdm import tqdm



import warnings



global moveOptions

# step correspoinding to grid neighbour

moveOptions = np.array([[0,-1],[0,1],[1,0],[-1,0]])
def plot_line(line,n,m):

    lc = mc.LineCollection(line, colors = [0,0,0,1], linewidths = 0.5)

    fig, ax = plt.subplots(dpi=200)

    #set x/y limits to match grid size

    #x/y ticks for each coordinate 

    plt.setp(ax, xlim = (-0.5,n-0.5), ylim = (-0.5,m-0.5),

             xticks = [], yticks = [],

             xticklabels = [], yticklabels = [])

    ax.add_collection(lc)

    ax.grid()

    plt.savefig('longLine.jpg')
# i and j are the indices for the node whose neighbors you want to find

def find_neighbours(grid, i, j):

    return np.array([grid[i+1,j], grid[i-1,j],grid[i,j+1], grid[i,j-1]])
def generate_line(line,n,m):



    # grid is True is space is valid to be occupied, otherwise False

    # padded with False (boundaries)

    grid = np.ones((n,m),bool)

    pad = np.zeros((n+2,m+2),bool)



    # set initial coordinate

    if not line:

        # if no line to start from, start at 0,0

        coor = (0,0)

    else:

        # otherwise start at last coordinate of line

        coor = line[-1][-1]

        

        # mark invalid (already occupied) coordinates on grid

        start = line[0][0]

        grid[n-start[1]-1, start[0]] = False

        for segment in line:

            endSeg = segment[-1]

            grid[n-endSeg[1]-1, endSeg[0]] = False

    



        

        

    # run until no valid moves left

    while True:



        # mark coordinates that have been occupied

        # lineCollections plot inverted so must transform coordinates to match

        grid[n-coor[1]-1, coor[0]] = False



        # insert grid such that there is 1 cell width boundary of False

        pad[1:-1,1:-1] = grid



        neighbours = find_neighbours(pad, n-coor[1], coor[0]+1)

        validOptions = moveOptions[neighbours]



        if not np.any(neighbours):

            break



        move = np.random.randint(0,len(validOptions))

        nextCoor = list(coor)

        nextCoor += validOptions[move]

        nextCoor = tuple(nextCoor)



        line.append([coor,nextCoor])

        coor = nextCoor

        

    return line
def calculate_line_variance(line,segLength):

    i = 0

    segmentVarianceMeans = np.zeros(len(segLength))

    for segL in segLength:

        # segment line into segments of certain length

        segments = [line[i:i+segL] for i in range(len(line)-(segL-1))]



        # store segment variances

        segmentVariance = np.zeros(len(segments))





        for seg in segments:

            segmentVariance = np.var(seg)

            

        segmentVarianceMeans[i] = segmentVariance.mean()

        

    return segmentVarianceMeans.mean()
## PLAY WITH THESE PARAMETERS ##



#Grid Dimensions

n = 12

m = 12



#number of lines to compare in each training cycle

populationSize = 1*10**3



#number of training cycles

generations = 100



################################



bestFitness = np.zeros(generations)



 # create empty dataframe to populate

index = range(populationSize)

columns = ['line','length','variance']

data = pd.DataFrame(index=index, columns=columns)



varSegLength = np.array([3,8,12,24])



for gen in tqdm(range(generations)):



    # populate dataframe

    for i in range(populationSize):

        if gen == 0:

            line = generate_line([],n,m)

        else:

            # slice off end of line at random point 



            # max length cut is proportional to training generation 

            longestCut = (varSegLength.min() + 1) + int((gen/generations)*(len(bestLine)-1))

            # minimum cut is varSegLength

            cut = np.random.randint(varSegLength.min()-1,longestCut)



            #regrow line randomly

            line = generate_line(bestLine[:cut],n,m)



        length = len(line)

        with warnings.catch_warnings(): 

            warnings.simplefilter('ignore')

            lineVariance = -(calculate_line_variance(line,varSegLength))

        data.loc[i]['line'] = line

        data.loc[i]['length'] = length

        data.loc[i]['variance'] = lineVariance



    bestFitness[gen] = (data['length'] + data['variance']).max() 

    bestLine = data['line'][(data['length'] + data['variance']) == bestFitness[gen]].values[0]



    

#PLOT

plot_line(bestLine,n,m)

fig, ax = plt.subplots()

plt.plot(bestFitness,color = 'black')

plt.xlabel('Training Cycles')

plt.ylabel('Space Filling')

ax.set_xlim(0,generations)

plt.savefig('fitness.jpg')

plt.show()
sns.distplot(pd.to_numeric(data['length']))

plt.show()
sns.distplot(pd.to_numeric(-data['variance']))

plt.show()