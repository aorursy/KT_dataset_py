# this genetic algorithm is for optimizing the value of the function of -(x**2) +5



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
#initialize population

import random

best=-100000

populations =([[random.randint(0,1) for x in range(6)] for i in range(4)])

print(type(populations))

parents=[]

new_populations = []

print(populations)





#fitness score calculation ............

def fitness_score() :

    global populations,best

    fit_value = []

    fit_score=[]

    for i in range(4) :

        chromosome_value=0

        

        for j in range(5,0,-1) :

            chromosome_value += populations[i][j]*(2**(5-j))

        chromosome_value = -1*chromosome_value if populations[i][0]==1 else chromosome_value

        print(chromosome_value)

        fit_value.append(-(chromosome_value**2) + 5 )

    print(fit_value)

    fit_value, populations = zip(*sorted(zip(fit_value, populations) , reverse = True))

    best= fit_value[0]

    

fitness_score()

#print(type(populations))

#selecting parents....

def selectparent():

    global parents

    #global populations , parents

    parents=populations[0:2]

    print(type(parents))

    print(parents)

selectparent()
#single-point crossover .........



def crossover() :

    global parents

    

    cross_point = random.randint(0,5)

    parents=parents + tuple([(parents[0][0:cross_point +1] +parents[1][cross_point+1:6])])

    parents =parents+ tuple([(parents[1][0:cross_point +1] +parents[0][cross_point+1:6])])

    

    print(parents)

    



crossover()
def mutation() :

    global populations, parents

    mute = random.randint(0,49)

    if mute == 20 :

        x=random.randint(0,3)

        y = random.randint(0,5)

        parents[x][y] = 1-parents[x][y]

    populations = parents

    print(populations)

mutation()
for i in range(1000) :

    fitness_score()

    selectparent() 

    crossover()

    mutation()

print("best score :")

print(best)

print("sequence........")

print(populations[0])

    