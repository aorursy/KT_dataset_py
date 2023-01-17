%matplotlib notebook
import seaborn as sns # used for the heatmap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.animation as animation
grid = pd.DataFrame({i: [False] * (50) for i in range(122)}) # 120x50 grid of zeros/False
grid[59][0] = True

for t in range(0, 50): # time evolution for 50 generations (Erzeugungen, nicht Generationen), t goes from 0 to 49
    
    for i in range(1, 121): # i goes from 1 to 120
        grid[i][t+1] = grid[i-1][t] != (grid[i][t] or grid[i+1][t])

            
# a) plot
plt.figure()
plot = sns.heatmap(grid.iloc[0:51, 1:121], xticklabels=False, yticklabels=False, cbar=False)
plt.title("Rule 30")
plt.xlabel("sites")
plt.ylabel("time")
plt.show()
# b)
def n1(t): #time dependence of the number of cells with z_i = 1:
    return grid[t].sum()

t = 50
print('number of cells with z_i = 1 for t = ' + str(t) + ' is ' + str(n1(t)))
# !!! dataframe[column][row] BUT dataframe.iloc[i_th row : j_th row][i_th column : j_th column] !!!

j = 20 + 2 # variable for grid size, need 21 + 0th for 20x20 grid, because of the artificial periodic boundaries i create
t_max = 135 # number of generations
life_rule_1 = ['21', '30', '31'] # these give 1

# a)/b)
def periodic_bound(x): # creating artificial periodic boundaries, by copying first row below last row, ..
    y = x.copy()
    
    # edges
    y[0][0] = x[j-2][j-2]
    y[j-1][j-1] = x[1][1]
    y[0][j-1] = x[j-2][1]
    y[j-1][0] = x[1][j-2]
    
    for i in range(1, j-1): # first and last row and column without edges
        y[i][0] = x[i][j-2]
        y[i][j-1] = x[i][1]
        y[0][i] = x[j-2][i]
        y[j-1][i] = x[1][i]
    
    return y


# a)/b)
def evolution(x):
    y = x.copy()
    
    for col in range(1, j-1):
        for row in range(1, j-1):
            nz = str(x.iloc[row-1:row+2, col-1:col+2].sum().sum() - x[col][row]) + str(x[col][row])
            y[col][row] = int(nz in life_rule_1) # = 1 if nz is in life_rule_1, else 0
            
    return y

# c)
def n2(t): #time dependence of the number of cells with z_i = 1:
    return generations[t].iloc[1:j-1, 1:j-1].sum().sum() # 1st .sum() gives a col of sum of columns, 2nd .sum() gives sum of that col
# !!! dataframe[column][row] BUT dataframe.iloc[i_th row : j_throw][i_th column : j_th column] !!!

life_game = pd.DataFrame({i: [0] * (j) for i in range(j)}) # grid of zeros

# select starting configuration:
configuration = 5

if configuration == 1: # square
    life_game.iloc[int(j/2)-1:int(j/2)+1, int(j/2)-1:int(j/2)+1] = 1
elif configuration == 2: # cross
    life_game.iloc[int(j/2)-1:int(j/2), int(j/2)-2:int(j/2)+1] = 1
    life_game.iloc[int(j/2)-2:int(j/2)+1, int(j/2)-1:int(j/2)] = 1
    life_game[int(j/2)-1][int(j/2)-1] = 0
elif configuration == 3: # 3bar
    life_game.iloc[int(j/2)-1:int(j/2), int(j/2)-2:int(j/2)+1] = 1
elif configuration == 4: # tetris
    life_game.iloc[int(j/2)-1:int(j/2), int(j/2)-2:int(j/2)+1] = 1
    life_game.iloc[int(j/2)-2:int(j/2)-1, int(j/2)-1:int(j/2)+2] = 1
elif configuration == 5: # glider
    life_game.iloc[int(j/2)-1:int(j/2), int(j/2)-2:int(j/2)+1] = 1
    life_game[int(j/2)][int(j/2)-2] = 1
    life_game[int(j/2)-1][int(j/2)-3] = 1
elif configuration == 6: # diehard
    life_game.iloc[int(j/2):int(j/2)+1, int(j/2):int(j/2)+3] = 1 # bar
    life_game[int(j/2)+1][int(j/2)-2] = 1 # dot
    life_game[int(j/2)-4][int(j/2)] = 1 # edge
    life_game[int(j/2)-4][int(j/2)-1] = 1
    life_game[int(j/2)-5][int(j/2)-1] = 1

    
generations = [] # list that contains the generations
generations.append(life_game) # initial configurations

    
for t in range(0, t_max):
    temp = generations[t].copy()
    generations.append(evolution(periodic_bound(temp)))
# animated plot

fig = plt.figure()
data = generations[0].iloc[1:j-1, 1:j-1]
grid = sns.heatmap(data, xticklabels=False, yticklabels=False, cbar=False)

def init(): # initializes the animation, calls first frame
    plt.clf()
    grid = sns.heatmap(data, xticklabels=False, yticklabels=False, cbar=False)

def animate(i): # calls the i'th frame
    plt.clf()
    data = generations[i].iloc[1:j-1, 1:j-1]
    grid = sns.heatmap(data, xticklabels=False, yticklabels=False, cbar=False)
    plt.title('t = ' + str(i))

anim = animation.FuncAnimation(fig, animate, init_func=init, interval=100, frames=t_max, repeat=False) # interval is delay between frames in ms

plt.show()
# c)

t = 130 # has to be </= t_max
print('number of live cells for t = ' + str(t) + ' is ' + str(n2(t)))
N = 50 # number of sites
Nt = 100 # number of iterations

def comp_flow(x):
    flow = 0
    
    for t in range(Nt):
        new_config = x.copy()
        site = 0
        
        while site < N-1:
            if x[site] == 1 and x[site + 1] == 0:
                new_config[site] = 0
                new_config[site + 1] = 1
                site += 1 # cell can only move once per cycle, so next site has to be skipped
            
            site += 1
        
        if x[N-1] == 1 and x[0] == 0: # check if flow happens
            new_config[N-1] = 0
            new_config[0] = 1
            flow += 1
        
        x = new_config.copy()
        
    return flow


def rand_bin_array(x): # creats list with x amount of ones and rest is zero
    arr = np.zeros(N)
    arr[:x]  = 1
    np.random.shuffle(arr)
    return arr
# a)

config = rand_bin_array(25)
print('flow for ' + str(Nt) + ' timesteps and '+ str(int(config.sum()))+ ' starting live cells is ' + str(comp_flow(config)))
# b) 
flow_arr = []
M_arr = range(1, N+1)
dens_arr = [M/N for M in M_arr]
num_config = 20

for M in M_arr:
    flow_avrg = 0
    
    for i in range(num_config): # computing average flow
        flow_avrg += comp_flow(rand_bin_array(M))
        
    flow_arr.append(flow_avrg/num_config)
    
plt.figure()
plt.plot(dens_arr, flow_arr)
plt.xlabel('density')
plt.ylabel('flow')
plt.title('flow vs density')
plt.show()