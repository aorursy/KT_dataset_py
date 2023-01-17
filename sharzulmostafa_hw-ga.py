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

chrosome = []

def init_popu():

    for i in range(4):

        arr=[]

        for j in range(6):

            arr.append(random.randint(0,1))

        chrosome.append(arr)

    print(chrosome)

def fit(ch):

    val=0

    ck=0

    c_val=0

    for i in range(len(ch)):

        if(i==0):

            if(ch[i]==1):

                ck=1

        else:

            val+=math.pow(2,(len(ch)-1-i))*ch[i]

    if(ck==1):

        val=-val

    c_val=val

    val=-(val*val)+5

    print("Decimal Value ",c_val)

    print("Fit Value ",val)

    return val,c_val

def store_fit(ch):

    net_fit=[]

    ch_val_fit=[]

    for i in range(len(ch)):

        val,ch_val=fit(ch[i])

        net_fit.append(val)

        ch_val_fit.append(ch_val)

        

    return net_fit,ch_val_fit

def selection(fit_val):

    first_val=-999999999

    first_index=-1

    sec_val=-999999999

    sec_index=-1

    for i in range(len(fit_val)):

        if(fit_val[i]>=first_val):

            first_val=fit_val[i]

            first_index=i

    

    for i in range(len(fit_val)):

        if(fit_val[i]>=sec_val and i!=first_index):

            sec_val=fit_val[i]

            sec_index=i

    print("Best Fit ",first_val)

    return first_index,sec_index

def crossover(f,s):

    m=random.randint(1,len(f))

    for i in range (m,len(f)):

        f[i],s[i]=s[i],f[i]

    return f,s

def mutation():

    ran=random.randint(0,50)

    if(ran==20):

        i=random.randint(0,3)

        j=random.randint(0,5)

        chrosome[i][j]=1-chrosome[i][j]

def all():

    fit_val,ch_val=store_fit(chrosome)  

    

    f_index,sec_index=selection(fit_val)



    first=chrosome[f_index]

    second=chrosome[sec_index]

    first,second=crossover(first,second)

    chrosome[2]=first

    chrosome[3]=second

    mutation()

init_popu()

for it in range(1001):

    print("Iteration ",it)

    all()