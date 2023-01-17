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

import math

import random

import copy

chorm=[]

#new_fit=[]
save_current_best=-9999999999

save_best=-99999999999999
def init_population():

    for i in range(4):

        arr=[]

        for j in range(6):

            arr.append(random.randint(0,1))

        chorm.append(arr)

    print(chorm)
def fitness(s_chrom):

    val=0

    flag=0

    ch_val=0

    for i in range(len(s_chrom)):

        if i==0:

            if s_chrom[i]==1:

                flag=1

        else:

            val=val+math.pow(2,len(s_chrom)-i-1)*s_chrom[i]

    if flag==1:

        val=-val

    ch_val=val

    val=-(val*val)+5

    print("Decimal Value ",ch_val)

    print("Fit Value ",val)

    return val,ch_val
def cal_fitness(chrom):

    n_fit=[]

    ch_val_list=[]

    for i in range(len(chrom)):

        val, ch_val=fitness(chrom[i])

        n_fit.append(val)

        ch_val_list.append(ch_val)

    return n_fit, ch_val_list
def selection(fit_val):

    first_best=-999999999

    first_index=-1

    second_best=-999999999

    second_index=-1

    for i in range(len(fit_val)):

        if fit_val[i]>=first_best:

            first_best=fit_val[i]

            first_index=i

            

    for i in range(len(fit_val)):

        if i!=first_index and fit_val[i]>=second_best:

            second_best=fit_val[i]

            second_index=i

    return first_index,second_index
def crossover(first,second):

    mutation_point=random.randint(1,len(first))

    for i in range(mutation_point,len(first)):

        first[i],second[i]=second[i],first[i]

    return first, second
def mutation(chrom):

    r=random.randint(0,50)

    if(r==20):

        i=random.randint(0,len(chrom)-1)

        j=random.randint(0,len(chrom[i])-1)

        chrom[i][j]=1-chrom[i][j]
def genet_algo():

    fit_val,ch_val=cal_fitness(chorm)    

    f_index,sec_index=selection(fit_val)

    first=chorm[f_index]

    second=chorm[sec_index]

    first,second=crossover(first,second)

    chorm[2]=first

    chorm[3]=second

    mutation(chorm)

init_population()
for it in range(1001):

    print("Iteration ",it)

    genet_algo()