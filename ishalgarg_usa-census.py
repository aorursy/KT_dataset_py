# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
datafile = "../input/us-census-data/adult-training.csv"

data = pd.read_csv(datafile)
import logistic_regression as lr
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data.head()
X[:,1] = le.fit_transform(X[:,1])
X
X[:,3] = le.fit_transform(X[:,3])

X[:,5] = le.fit_transform(X[:,5])

X[:,6] = le.fit_transform(X[:,6])

X[:,7] = le.fit_transform(X[:,7])

X[:,8] = le.fit_transform(X[:,8])

X[:,9] = le.fit_transform(X[:,9])

X[:,13] = le.fit_transform(X[:,13])
X
X = np.array(X,dtype=np.int)
X
Y.shape
Y = Y.reshape(1,Y.shape[0])
for i in range(Y.shape[1]):

    if(Y[0][i] == '<=50k'):

        Y[0][i] = 0

    else:

        Y[0][i] = 1
Y
Y = np.array(Y,dtype=np.int)
a = lr.logisticRegerssion()
X  = a.normalize(X)
X = np.transpose(X)
parameters = a.parameters_intialize(X)
parameters
j_graph=[]

no_of_iter = []

for i in range(1000):

    caches = a.forward_activation(X,parameters)

    cost = a.cost(caches["A1"],Y)

    j_graph.append(cost)

    no_of_iter.append(i)

    caches_d = a.backward_prop(caches["A1"],Y,X)

    parameters = a.updation(parameters,caches_d,learning_rate = 0.3)

    print("Cost after " + str(i) + " iteration " + str(cost)) 
import matplotlib.pyplot as plt
plt.plot(j_graph,no_of_iter)
A=caches["A1"]

for i in range(A.shape[1]):

    if(A[0][i]>=0.5):

        A[0][i] = 1

    else:

        A[0][i] = 0
accuracy = 0

for i in range(A.shape[1]):

    if(A[0][i]==Y[0][i]):

        accuracy+=1



m = Y.shape[1]

accuracy=(accuracy*100)/m
accuracy
data_train = "../input/us-census-data/adult-training.csv"

data_test = pd.read_csv(data_train)
X = data_test.iloc[:,:-1].values

Y = data_test.iloc[:,-1].values

data_test.head()
X[:,1] = le.fit_transform(X[:,1])

X[:,3] = le.fit_transform(X[:,3])

X[:,5] = le.fit_transform(X[:,5])

X[:,6] = le.fit_transform(X[:,6])

X[:,7] = le.fit_transform(X[:,7])

X[:,8] = le.fit_transform(X[:,8])

X[:,9] = le.fit_transform(X[:,9])

X[:,13] = le.fit_transform(X[:,13])
X = np.array(X,dtype=np.int)
Y = Y.reshape(1,Y.shape[0])

for i in range(Y.shape[1]):

    if(Y[0][i] == '<=50k'):

        Y[0][i] = 0

    else:

        Y[0][i] = 1

        

Y = np.array(Y,dtype=np.int)
X = a.normalize(X)
X = np.transpose(X)

caches = a.forward_activation(X,parameters)
cost = a.cost(caches["A1"],Y)
accuracy = 0

for i in range(A.shape[1]):

    if(A[0][i]==Y[0][i]):

        accuracy+=1



m = Y.shape[1]

accuracy=(accuracy*100)/m
accuracy
cost