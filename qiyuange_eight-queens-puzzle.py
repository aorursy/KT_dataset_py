from IPython.display import Image
import os
Image(r'../input/8-queens-images/8 Queens images/sample.jpg')
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(271)
sns.set_style("whitegrid")
Image(r'../input/8-queens-images/8 Queens images/1.PNG')
def softmax(input):
    input = np.array(input, dtype=np.float)
    input = np.exp(input)
    output = input / input.sum()
    return output
def fitness_function(individual):
    value = 0
    for i in range(7):
        for j in range(i+1,8,1):
            if individual[i] != individual[j]:
                x_distance = np.abs(individual[j] - individual[i])
                y_distance = j - i
                if x_distance != y_distance:
                    value += 1
    return value
def mutation(individual, prob=0.1):
    p = np.random.rand(8)
    individual[p>prob] = np.random.choice(range(8), 8)[p>prob]
    
    return individual
def GA(size = 4):
    size = size
    num_generation = 0
    population = []
    for i in range(size):
        population.append(np.random.choice(range(8), 8))
    while (True):
        print("Generation : ", num_generation)
        fitness_list = []
        selection = []
        
        for individual in population:
            fitness_value = fitness_function(individual)
            if fitness_value == 28:
                print("Find Target!")
                print(individual)
                return individual
            fitness_list.append(fitness_value)
        
        print(fitness_list)
        print()
        
        #Selection is Here
        prob = softmax(fitness_list)
        select_id = np.random.choice(range(size), size, replace=True, p=prob)
        for idx in select_id:
            selection.append(population[idx])
        num_pair = int(size/2)
        position = np.random.choice(range(1,7,1), num_pair, replace=True)
        
        
        #Crossover is Here
        for i in range(0, size, 2):
            start = position[int(i/2)]
            tempa = copy.deepcopy(selection[i][start:])
            tempb = copy.deepcopy(selection[i+1][start:])
            selection[i][start:] = tempb
            selection[i+1][start:] = tempa
            
            
        #Mutation is Here
        for i in range(size):
            selection[i] = copy.deepcopy(mutation(selection[i], prob=0.8))
        population = selection
        num_generation += 1
def display(input):
    matrix = np.zeros((8,8))
    for i in range(8):
        matrix[i][input[i]] = 1.0
    return matrix
Queen = GA(size = 4)
image = display(Queen)
image
plt.figure(figsize=(8,8))
plt.imshow(image, cmap='gray')
Queen = GA(size = 100)
image = display(Queen)
image
plt.figure(figsize=(8,8))
plt.imshow(image, cmap='gray')
Queen = GA(size = 1000)
image = display(Queen)
image
plt.figure(figsize=(8,8))
plt.imshow(image, cmap='gray')