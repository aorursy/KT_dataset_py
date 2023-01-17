#KPCA for d=3

import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt
#read dataset

data=pd.read_csv('../input/prmlassignment/Dataset3.csv',header=None)

data.head()
#store data in x and y

x=data[0]

y=data[1]



x=list(map(float,x))

y=list(map(float,y))
#find covariance matrix covar for polynomial kernel where d=3

covar=[[0.0 for j in range(0,1000)]for i in range(0,1000)]

for i in range(0,1000):

    for j in range(0,1000):

        covar[i][j]=(x[i]*x[j]+y[i]*y[j]+1)*(x[i]*x[j]+y[i]*y[j]+1)*(x[i]*x[j]+y[i]*y[j]+1)
#finding eigen values and eigen vectors

from numpy.linalg import eig

evalues,evectors=eig(covar)

evalues=evalues.real

evectors=evectors.real

evalues=list(evalues)

evectors=list(evectors)

#print(evalues,evectors)

firstl=evalues[0]

secondl=evalues[1]

first_eval=[]

second_eval=[]

for i in range(0,1000):

    first_eval.append(evectors[i][0])

    second_eval.append(evectors[i][1])

for i in range(0,1000):

    first_eval[i]=first_eval[i]*(1.0/(math.sqrt(1000*firstl)))

    second_eval[i]=second_eval[i]*(1.0/(math.sqrt(1000*secondl)))
data_x1=[]

data_x2=[]

for i in range(0,1000):

    temp_res1=0.0

    temp_res2=0.0

    for j in range(0,1000):

        temp_res1+=(first_eval[j]*(x[i]*x[j]+y[i]*y[j]+1)*(x[i]*x[j]+y[i]*y[j]+1)*(x[i]*x[j]+y[i]*y[j]+1))

        temp_res2+=(second_eval[j]*(x[i]*x[j]+y[i]*y[j]+1)*(x[i]*x[j]+y[i]*y[j]+1)*(x[i]*x[j]+y[i]*y[j]+1))

    data_x1.append(temp_res1)

    data_x2.append(temp_res2)
plt.plot(data_x1,data_x2,'.g')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.show()