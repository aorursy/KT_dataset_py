import numpy as np



import os

import json



import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.animation import ArtistAnimation

from matplotlib import colors



from IPython.display import Image, display



from scipy.ndimage import convolve



from tqdm.notebook import tqdm 



global cmap

global norm



cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)
def plot_task(task):

    trainInput = taskTrain[task][0]['input']

    trainOutput = taskTrain[task][0]['output']

    

    testInput = taskTest[task][0]['input']

    testOutput = taskTest[task][0]['output']

    

    fig, axs = plt.subplots(1,4, figsize=(12,6))

    

    plt.setp(axs, xticks = [], yticks =[], xticklabels=[], yticklabels=[] )

    

    plt.suptitle(taskFileNames[task])

    

    axs[0].imshow(trainInput,cmap=cmap,norm=norm)

    axs[1].imshow(trainOutput,cmap=cmap,norm=norm)

    axs[2].imshow(testInput,cmap=cmap,norm=norm)

    axs[3].imshow(testOutput,cmap=cmap,norm=norm)

    

    axs[0].set_title('Train Input 0')

    axs[1].set_title('Train Output 0')

    axs[2].set_title('Test Input 0')

    axs[3].set_title('Test Output 0')

    

    plt.show()



# load in task files

evalPath = '/kaggle/input/abstraction-and-reasoning-challenge/training/'

taskTrain = list(np.zeros(len(os.listdir(evalPath))))

taskFileNames = list(np.zeros(len(os.listdir(evalPath))))

taskTest = list(np.zeros(len(os.listdir(evalPath))))



for i,file in enumerate(os.listdir(evalPath)):

    with open(evalPath + file, 'r') as f:

        task = json.load(f)

        taskFileNames[i] = file

        taskTrain[i] = []

        taskTest[i] = []

        

        for t in task['train']:

                 taskTrain[i].append(t)

        for t in task['test']:

                taskTest[i].append(t)

        

# plot 5 random tasks as examples

for i in np.random.randint(0,len(taskTrain),5):

    plot_task(i)
n = 100

m = 200



# create grid and set center cell in first row to alive as initial condition

grid = np.zeros((n,m))

grid[0,len(grid[0])//2] = 1



# rule 30 kernerls

rules = [[1,0,0],[0,1,1],[0,1,0],[0,0,1]]



fig, ax = plt.subplots(1,1, figsize = (12,6))

plt.setp(ax, xticklabels = [], xticks = [], yticklabels = [], yticks = [])



images = []



for i in range(len(grid)-1):

    for rule in rules:

        

        # apply each rule as a convolution and 

        convRule = convolve(grid[i], rule, mode = 'constant')

        conv = convolve(grid[i], [1,1,1], mode = 'constant')

        

        # compare only non-zero values

        conv[conv == 0] = -1

        

        # rule is true where conv == convRule

        grid[i+1][np.equal(conv,convRule)] = 1

        

        

    images.append([plt.imshow(grid, cmap = 'gray')])



plt.close()
# save movie to file

ani = ArtistAnimation(fig, images)

ani.save('Rule30.gif', writer='imagemagick', fps = 8 )
n = 200

m = 200

grid = np.random.randint(0,2,(n,m))



# kernel to get neighbours, excluding cell being 'looked at'

k = [[1,1,1],

     [1,0,1],

     [1,1,1]]



fig, ax = plt.subplots(1,1, figsize = (12,6))

plt.setp(ax, xticklabels = [], xticks = [], yticklabels = [], yticks = [])

images = []

steps = 100



for step in range(steps):



    # apply convolution to find neighbour count

    conv = convolve(grid, k, mode = 'constant')



    newGrid = np.zeros((n,m))



    # apply GOL rules

    newGrid[conv == 3] = 1

    newGrid[np.logical_and((grid == 1), (conv == 2))] = 1



    grid = newGrid



    images.append([plt.imshow(grid, cmap = 'gray')])

   

plt.close()
# save movie to file

ani = ArtistAnimation(fig, images)

ani.save('GOL.gif', writer='imagemagick', fps = 8 )
def create_rules(dna):

    

    dna = dna.reshape(len(dna)//13,13)

    

    # randomize order of gene rule application

    np.random.shuffle(dna)

    

    def rules(grid,output):

        

        for gene in dna:

            

            # binary 3x3 kernel indicating state of neighbours

            kernel = np.array([gene[:3],gene[3:6],gene[6:9]])

            

            # colours involved and threshold

            ruleColour = gene[9]

            oldColour = gene[10]

            newColour = gene[11]

            threshold = gene[12]

            

            # get binary grid indicating whether cell is ruleColour or not

            gridRuleColour = (grid == ruleColour).astype(int)

            

            c = convolve(gridRuleColour, kernel, mode='constant')

            

            

            # get boolean matrix of cells that satify rule condition

            rule = np.logical_and(c > threshold, grid == oldColour)

            

            output[rule] = newColour

            

        return output

    return rules 
import ga_utils as ga

import arc_utils as arc
def solve_task(task, taskTrain, taskTest, sameShape, nGenes,gens,steps,mutationRate):



    # count number of training examples and tests

    nExamples = len(taskTrain[task])

    nTests = len(taskTest[task])

    

    # empty list to be filled with best rules for each step

    bestSteps = [0]*steps

    

    for step in range(steps):

        

        best = 0

        bestCount = 0

        

        fitness = np.zeros((popSize,nExamples + 1))

        fitMean = np.zeros(gens)

        fitMax = np.zeros(gens)

        



        population = ga.create_population(popSize,nGenes)

        

        for g in tqdm(range(gens)):

            

            for i in range(len(population)):

                # create rules from each dna in the population

                rules = create_rules(population[i])



                # update grid for each training input example and calculate average fitness

                for j in range(nExamples):

                    

                    taskInput = np.array(taskTrain[task][j]['input'])

                    taskOutput = np.array(taskTrain[task][j]['output'])



                    

                    if not sameShape:

                        # "scale" matrix size to match output, n and m are scaling factors

                        n = taskOutput.shape[0]//taskInput.shape[0]

                        m = taskOutput.shape[0]//taskInput.shape[0]

                        taskInput = np.kron(taskInput,np.ones((n,m)))

  

                    

                    grid = taskInput

                                       

                    if step is not 0:

                        # update input with best rules for each steps

                        for s in range(step):

                            grid = arc.update_grid(grid,bestSteps[s])



                    grid = arc.update_grid(grid,rules)

                    

    

                    # calculate fitness after update steps

                    

                    fitness[i,j] = arc.calc_fitness(grid,taskInput,taskOutput)

                



                fitness[i,-1] = fitness[i,:-1].mean()



                # if we find the optimal solution (fitness score of 1)

                if (fitness[i,-1] >= best):

                    

                    best = fitness[i,-1]

                    bestSteps[step] = rules

                    

                    if best == 1:

                        

                        # find three optimal solutions before terminating

                        bestCount = bestCount + 1

                        

                        if bestCount == 1000:

                            fitMean[g] = fitness[:i+1,-1].mean()

                            fitMax[g] = 1



                            print('Optimal Solution Found for Task: ' + str(task))

                            arc.plot_evolve(taskFileNames[task],fitMean,fitMax)

                            arc.plot_solve(task, taskTrain, taskTest, sameShape, bestSteps, step + 1)

                            break

                        



            else:

                # if no break in inner loop

                

                if g is not (gens - 1):

                    sel = ga.selection(population,fitness[:,-1])

                    population = ga.reproduce(population,sel,geneLength,mutationRate)

                fitMean[g] = fitness[:,-1].mean()

                fitMax[g] = fitness[:,-1].max() 

                continue

            break

            

        # after the best rules for each step has been found

        # append function to bestSteps

        



        

        # kill off and replace bottom portion of population

        #population[len(population)//10:] = create_population(len(population)//10*9,nGenes)

        

        

        else:

            bestIndex = np.where(fitness[:,-1] == fitness[:,-1].max())[0][0]

            bestSteps[step] = create_rules(population[bestIndex])

            # if no break in inner loop

            arc.plot_evolve(taskFileNames[task], fitMean,fitMax)

            arc.plot_solve(task, taskTrain, taskTest, sameShape, bestSteps, step + 1)

            continue

        break

            

    # return best rules for each step

    return bestSteps
solutions = []



#blue and grey squares, blue and blue corner square, blue and red lines, blue and grey tetris

tasks = ['b60334d2.json','3aa6fb7a.json','a699fb00.json','3618c87e.json']



for i in range(len(tasks)):

    tasks[i] = np.where(np.array(taskFileNames) == tasks[i])[0][0]

    

for task in tasks:

    sameShape, sameColours = arc.check_task(task, taskTrain)

    

    popSize = 5000

    nGenes = 20

    gens = 30

    steps = 1

    geneLength = 12

    mutationRate = 1 



    try:

        print('Solving task: ' + taskFileNames[task])

        bestSteps = solve_task(task, taskTrain, taskTest, sameShape, nGenes, gens, steps, mutationRate)

        solutions.append([task, bestSteps])

    except:

        solutions.append([task,'Failed'])
#task = ['db3e9e38.json'] #orange and blue triangle

task = ['a65b410d.json'] #b/r/g stairs

task = np.where(np.array(taskFileNames) == task)[0][0]

    

sameShape, sameColours = arc.check_task(task, taskTrain)

    

popSize = 5000

nGenes = 30

gens = 30

steps = 5

geneLength = 12

mutationRate = 2 





print('Solving task: ' + taskFileNames[task])

bestSteps = solve_task(task, taskTrain, taskTest, sameShape, nGenes, gens, steps, mutationRate)

solutions.append([task, bestSteps])