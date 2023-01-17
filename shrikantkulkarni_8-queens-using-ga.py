# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import random
import numpy as np
from numpy.random import choice
import pandas as pd
def initial_solutions(pop):
    secure_random = random.SystemRandom()
    totalPopulation = pop
    populationData = []
    fitnessData = []
    secure_random = random.SystemRandom()
    for outloop in range(totalPopulation):
        randomData = []
        for inloop in range(8):
            selectedData = secure_random.choice([0,1,2,3,4,5,6,7])
            randomData.append(selectedData)
        populationData.append(randomData)
        fitnessData.append(fitment(randomData))
    probDataFrame = pd.DataFrame({'String':populationData,'FitnessScore':fitnessData})
    probDataFrame = probDataFrame.sort_values(['FitnessScore'],ascending=True)
    probDataFrame = probDataFrame.reset_index(drop=True)
    return probDataFrame, populationData, fitnessData
def fitment(array):
    board = np.zeros(shape=(8, 8))
    #example = [0, 7, 0, 3, 1, 2, 2, 4]
    for i in range(8):
        board[array[i],i]=1   
    row_count=0
    fitness_score = 0
    #check the columns and rows
    for row_num in range(8):
        row_count = 0
        for col_num in range(8):
            row_count = board[row_num,col_num] + row_count
        if row_count > 1 :
            fitness_score = fitness_score + int(row_count)
    #print (fitness_score)
    fitness_score = fitness_score + diaganoal_check(board)
    #print (fitness_score)
    return fitness_score
def return_board(array):
    board = np.zeros(shape=(8, 8))
    #example = [0, 7, 0, 3, 1, 2, 2, 4]
    for i in range(8):
        board[array[i],i]=1
    return board
def diaganoal_check(board):
#Left to right diagonal
    diagonal_score = 0
    if (board[6,0]+board[7,1]) > 1 :
        diagonal_score = diagonal_score +1
    if (board[5,0]+board[6,1]+board[7,2]) > 1:
        diagonal_score = diagonal_score +1
    if (board[4,0]+board[5,1]+board[6,2]+board[7,3]) > 1:
        diagonal_score = diagonal_score +1
    if (board[3,0]+board[4,1]+board[5,2]+board[6,3]+board[7,4]) > 1:
        diagonal_score = diagonal_score +1
    if (board[2,0]+board[3,1]+board[4,2]+board[5,3]+board[6,4]+board[7,5]) > 1:
        diagonal_score = diagonal_score +1
    if (board[1,0]+board[2,1]+board[3,2]+board[4,3]+board[5,4]+board[6,5]+board[7,6]) > 1:
        diagonal_score = diagonal_score +1
    if (board[0,0]+board[1,1]+board[2,2]+board[3,3]+board[4,4]+board[5,5]+board[6,6]+board[7,7]) > 1:
        diagonal_score = diagonal_score +1
    if (board[0,1]+board[1,2]+board[2,3]+board[3,4]+board[4,5]+board[5,6]+board[6,7]) > 1:
        diagonal_score = diagonal_score +1
    if (board[0,2]+board[1,3]+board[2,4]+board[3,5]+board[4,6]+board[5,7]) > 1:
        diagonal_score = diagonal_score +1
    if (board[0,3]+board[1,4]+board[2,5]+board[3,6]+board[4,7]) > 1:
        diagonal_score = diagonal_score +1
    if (board[0,4]+board[1,5]+board[2,6]+board[3,7]) > 1:
        diagonal_score = diagonal_score +1
    if (board[0,5]+board[1,6]+board[2,7]) > 1:
        diagonal_score = diagonal_score +1
    if (board[0,6]+board[1,7]) > 1:
        diagonal_score = diagonal_score +1
#Right to left 
    if (board[0,1]+board[1,0]) > 1:
        diagonal_score = diagonal_score +1
    if (board[0,2]+board[1,1]+board[2,0]) > 1:
        diagonal_score = diagonal_score +1
    if (board[0,3]+board[1,2]+board[2,1]+board[3,0]) > 1:
        diagonal_score = diagonal_score +1
    if (board[0,4]+board[1,3]+board[2,2]+board[3,1]+board[4,0]) > 1:
        diagonal_score = diagonal_score +1
    if (board[0,5]+board[1,4]+board[2,3]+board[3,2]+board[4,1]+board[5,0]) > 1:
        diagonal_score = diagonal_score +1
    if (board[0,6]+board[1,5]+board[2,4]+board[3,3]+board[4,2]+board[5,1]+board[6,0]) > 1:
        diagonal_score = diagonal_score +1
    if (board[0,7]+board[1,6]+board[2,5]+board[3,4]+board[4,3]+board[5,2]+board[6,1]+board[7,0]) > 1:
        diagonal_score = diagonal_score +1
    if (board[1,7]+board[2,6]+board[3,5]+board[4,4]+board[5,3]+board[6,2]+board[7,1]) > 1:
        diagonal_score = diagonal_score +1
    if (board[2,7]+board[3,6]+board[4,5]+board[5,4]+board[6,3]+board[7,2]) > 1:
        diagonal_score = diagonal_score +1
    if (board[3,7]+board[4,6]+board[5,5]+board[6,4]+board[7,3]) > 1:
        diagonal_score = diagonal_score +1
    if (board[4,7]+board[5,6]+board[6,5]+board[7,4]) > 1:
        diagonal_score = diagonal_score +1
    if (board[5,7]+board[6,6]+board[7,5]) > 1:
        diagonal_score = diagonal_score +1
    return diagonal_score
#This is execution
from datetime import datetime
target = 0
crossOverPoint = 4
generationCount = 1000
initial_population = 15
secure_random = random.SystemRandom()
#Initialize solutions
(probDataFrame, populationData, fitnessData) = initial_solutions(initial_population)
#print("Random Best solution1: " + str(probDataFrame[0:1]["String"].values[0]) + " with score " + str(probDataFrame[0:1]["FitnessScore"].values[0]))
#print("Random Best solution2: " + str(probDataFrame[1:2]["String"].values[0]) + " with score " + str(probDataFrame[1:2]["FitnessScore"].values[0]))
print(probDataFrame)
timeNow = datetime.now()
print("Starting the Genetic Computing:" + str(timeNow))
#Start GA with crossover and mutation
for loop in range(generationCount):
    boards=[]
    boards.append(probDataFrame[0:1]["String"].values[0])
    boards.append(probDataFrame[1:2]["String"].values[0])
    #print(boards)
    parent1_score = fitment(boards[0])
    parent2_score = fitment(boards[1])
    if (parent1_score==target or parent2_score==target ):
        print('winner', return_board(boards[0]), ' ',return_board(boards[1]))
        break
    #crossover by creating 2 variants from parents
    child1 = boards[0][0:crossOverPoint]+boards[1][crossOverPoint:]
    child2 = boards[1][0:crossOverPoint]+boards[0][crossOverPoint:]
    #Mutating one random digit
    child1[random.randint(0,7)] = secure_random.choice([0,1,2,3,4,5,6,7])
    child2[random.randint(0,7)] = secure_random.choice([0,1,2,3,4,5,6,7])
    #print('loop :',loop,' Child 1 : ' , child1, fitment(child1), ' Chilld2 : ', child2, fitment(child2))
    child1_score = fitment(child1)
    child2_score = fitment(child2)
    #if ( child1_score < parent1_score or  child1_score < parent2_score) :
    #   populationData.append(child1)
    #   fitnessData.append(child1_score)
    #   print('loop :',loop,'Appending Child 1 : ' ,child1, child1_score)
    #if ( child2_score < parent1_score or child2_score < parent2_score) :
    #   populationData.append(child1)
    #   fitnessData.append(child1_score)
    #   print('loop :',loop,'Appending Child 2 : ' ,child2, child2_score)    
    
    populationData.append(child1)
    fitnessData.append(child1_score)
    #print('loop :',loop,'Appending Child 1 : ' ,child1, child1_score)  
    populationData.append(child2)
    fitnessData.append(child2_score)
    #print('loop :',loop,'Appending Child 2 : ' ,child2, child1_score)  
    probDataFrame = pd.DataFrame({'String':populationData,'FitnessScore':fitnessData})
    probDataFrame = probDataFrame.sort_values(['FitnessScore'],ascending=True)
    probDataFrame = probDataFrame.reset_index(drop=True)

timeNow = datetime.now()
print("Ending the Genetic Computing:" + str(timeNow) +' after loops : ' + str(loop))
print("Best solution: " + str(boards[0]) + "Best score: " + str(parent1_score) )
return_board([3, 7, 0, 2, 5, 1, 6, 4])
fitment([3, 7, 0, 2, 5, 1, 6, 4])