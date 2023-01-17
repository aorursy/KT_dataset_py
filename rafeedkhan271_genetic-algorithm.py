 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random







import os

print(os.listdir("../input"))







chrom=[]

cond_val=2

mx=-100000
for i in range(4):

    array=[]

    for j in range(6):

        array.append(random.randint(0,1))

    chrom.append(array)    

    print(chrom[i])
def func(sm):

    #print(sm)

    ttt= - (sm*sm)+5

    return ttt
def fit_value():

    val=[]

    temp=[]

    for i in range(4):

        sm=0

        flag=0

        for j in range(6):

            if chrom[i][j]==1:

                if j==0:

                    flag=1

                elif j==1:

                    sm = sm+16

                elif j==2:

                    sm = sm+8

                elif j==3:

                    sm = sm+4

                elif j==4:

                    sm = sm+2

                elif j==5:

                    sm = sm+1

        if flag==1:

            sm=-sm

            

        sm=func(sm)    

        val.append(sm)

        temp.append(sm)

    #print(temp)

    val.sort()

    #print(val)

    for i in range(4):

        if temp[i]==val[3]:

            f=i

        if temp[i]==val[2]:

            s=i

        

    return (val[3],val[2],f,s)

    

        
def crossover(x,y):

    new_chrom=[]

    temp=random.randint(0,5)

    #print(temp)

    new_chrom.append(x)

    new_chrom.append(y)

    #print(new_chrom)

    for i in range(2):

        array=[]

        for j in range(6):

            if j<=temp:

                array.append(new_chrom[i][j])

            else:    

                array.append(new_chrom[1-i][j])

        new_chrom.append(array)    

    

    #print(new_chrom)

    return new_chrom
def mutation():

    cond=random.randint(1,50)

    #print(cond)

    #print(cond_val)

    if cond==cond_val:

        temp=random.randint(0,3)

        t=random.randint(0,5)

        return(temp,t)

    else:

        return(-1,-1)
def best_value():

    val=[]

    temp=[]

    for i in range(4):

        sm=0

        flag=0

        for j in range(6):

            if chrom[i][j]==1:

                if j==0:

                    flag=1

                elif j==1:

                    sm = sm+16

                elif j==2:

                    sm = sm+8

                elif j==3:

                    sm = sm+4

                elif j==4:

                    sm = sm+2

                elif j==5:

                    sm = sm+1

       

        if flag==1:

            sm=-sm

        sm=func(sm)    

        val.append(sm)

    val.sort()

    return val[3]

for i in range(500):

    #print(chrom)

    first,second,x,y=fit_value()

    #print(first, second)

    #temp_chro

    x,y=chrom[x],chrom[y]

    #print(chrom)

    chrom=crossover(x,y)

    #print(chrom)

    temp,t=mutation()

    #print(temp,t)

    if temp!=-1 and t!=-1:

        tt=chrom[temp][t]

        chrom[temp][t]=1-tt   

    #print(chrom)

    ans=best_value()

    mx=max(ans,mx)

    print("Iteration=",i,", Best value=",mx)

    

    