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

def genChromosome():

    for i in range(4):

        sChromosome = []

        for j in range(6):

            sChromosome.append(random.randint(0,1))

        chromosome.append(sChromosome)

        

    print (chromosome)

genChromosome()
def evadecimal():

    fitval.clear()

    for i in range(4):

        val=0

        for j in range(1,6):

            val += (math.pow(2,5-j))*chromosome[i][j]

        if chromosome[i][0] == 1:

            val= -val

        fitval.append(val)

    print("New fitval=",fitval)



            

            
evadecimal()
def newdic():

    newfitval = {}

    a=-1

    b=-1

    for i in range(4):

        newfitval[np.abs(5-fitval[i])]=i

    newfitval=dict(sorted(newfitval.items()))



    ch=0

    for key in newfitval:

        ch=ch+1

        if ch==0:

            b=newfitval[key]

            continue   

        if ch==1:

            a=newfitval[key]

            print("Bestval=",key)

            break

            

    return a,b



print(newdic())





            

        

            

        
def crossover():

    select = random.randint(0,5)

    a,b=newdic()

    for i in range(select+1,6):

        chromosome[a][i],chromosome[b][i]=chromosome[b][i], chromosome[a][i]

    print("Selected ",select,chromosome)

    

crossover()
def mutation():

    select = random.randint(1,50)

    if select==25:

        select2 = random.randint(0,3)

        select3 = random.randint(0,5)

        chromosome[select2][select3]=1- chromosome[select2][select3]

        print("Mutation at",select2,select3)

        



        

        

        
genChromosome()

for i in range (1000):

    print("Ita",i)

    evadecimal()

    crossover()

    mutation()