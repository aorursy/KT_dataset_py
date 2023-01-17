# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
chromosome = []

def popu_initialization():    

    for i in range(4):

        array = []      

        for j in range(6):

            array.append(random.randint(0,1))

            

        chromosome.append(array)

        #print(chromose)
def fitness_calculation(single_chrom):   

    flag = 0

    val = 0   

    for i in range(6):

        if i == 0 :

            flag = 1

        else:

            val += single_chrom[i]* pow(2,(5-i))

            

    if flag == 1:

        val = -val

        

    return val


def fitness(chrom):

    for i in range(4):

        val_with_index = []

        val = fitness_calculation(chrom[i])

        #print(chrom[i])

        #print(val)

        val = - (val * val) + 5

        val_with_index.append(val)

        val_with_index.append(i)

        #print(val_with_index)

        value.append(val_with_index)

        

        
def crossover(best_index, scnd_best_index):

    flag = 0

    r = random.randint(1,6)

    best_chrom = chromosome[best_index]

    scnd_best_chrom = chromosome[scnd_best_index]

    print(best_chrom,scnd_best_chrom)

    for i in range(4):

        if i != best_index and i != scnd_best_index :

            for j in range(6):

                if flag == 1:

                    if j > r :

                        chromosome[i][j] = best_chrom[j]

                    else :

                        chromosome[i][j] = scnd_best_chrom[j]

                    

                else:

                    if j > r :

                        chromosome[i][j] = scnd_best_chrom[j]

                    else :

                        chromosome[i][j] = best_chrom[j]

                

            flag = 1

    

    #print(r)

    #print(chromosome)
def mutation(prb):

    r = random.randint(1,(100/prb))

    if r == 20:

        rand_chrom = random.randint(0,3)

        rand_bit = random.randint(0,5)

        chromosome[rand_chrom][rand_bit] = 1 - chromosome[rand_chrom][rand_bit]
popu_initialization()

for t in range(500):

    value = []

    fitness(chromosome)

    #print(chromosome)

    value = sorted(value)

    #print(value)



    best_index = value[3][1]

    print(value[3][0])

    scnd_best_index = value[2][1]



    #print(best_index,scnd_best_index)



    crossover(best_index,scnd_best_index)

    prb = 2

    mutation(prb)