# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import random

import math

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
chromosome = []

fitval = []

def generateCromosome():

    chromosome.clear()

    for i in range(4):

        singleChromosome = []

        for j in range(6):

            singleChromosome.append(random.randint(0,1))

        chromosome.append(singleChromosome)



    print ("Generated chromosome = ",chromosome)

def evaluateSolution():

    fitval.clear()

    for i in range(4):

        val = 0

        for j in range(1,6):

            val += math.pow(2,5-j)*chromosome[i][j]

        if chromosome[i][0] == 1:

            val = - val

        fitval.append(val)

    print ("Fitval = ",fitval)

def func(x):

    return -(x*x)+5
def selection():

    ftval2 = [[0 for i in range(2)] for j in range(4)]

    print (ftval2)

    c1=-1

    c2=-1

    for i in range(4):

        ftval2[i][0]=func(fitval[i])

        ftval2[i][1]=i

        

    ftval2=sorted(ftval2,key=lambda l:l[0], reverse=True)

    

    c1=ftval2[0][1]

    c2=ftval2[1][1]

    

    bstval = ftval2[0][0]

    print("Bestval = ",bstval)

    return c1, c2, bstval

def crossover(s1,s2):

    select = random.randint(0,5)

    

    for i in range(select, 6):

        chromosome[s1][i],chromosome[s2][i] = chromosome[s2][i],chromosome[s1][i]

    print("For crossover, Selected = ",select,", Chromose no = ",c1,c2,", After Crossover = ",chromosome)

def mutation():

    select = random.randint(1,50)

    if select == 30:

        select2 = random.randint(0,3)

        select3 = random.randint(0,5)

        chromosome[select2][select3] = 1 - chromosome[select2][select3]

        print("Mutation occured at, Chromosome = ",select2,", Position = ",select3)

    
generateCromosome()

for i in range(1000):

    print("Iteration",i+1)

    evaluateSolution()

    c1,c2,bstval=selection()

    if bstval==5.0:

        break

    crossover(c1,c2)

    mutation()


