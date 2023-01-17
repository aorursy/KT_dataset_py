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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data = pd.read_csv("../input/data.csv")

# Normalize the data

data = (data - data.mean())/(data.std()) 

data.insert(0,"ones",1)

print(data.head(10))

####

cols = data.shape[1]

x=np.matrix(data.iloc[:,:cols-1])

y=np.matrix(data.iloc[:,cols-1:cols])

theta = np.matrix(np.zeros(cols-1))

####

def cost(X,Y,Theta):

    error = np.sum(np.power(((X * Theta.T)-Y),2)) / (len(X)*2)

    return error

print("COST = ",cost(x,y,theta))

####

def GD(X,Y,Theta,Alphe,Iters):

    parmaters = Theta.shape[1]

    g = np.matrix(np.zeros(parmaters))

    cost_list = []

    for i in range(Iters):

        error = (X * Theta.T)-y

        for j in range(parmaters):

            term   = np.multiply(error,X[:,j])

            g[0,j] = Theta[0,j] - ((Alphe / len(x)) * np.sum(term))

        Theta = g

        cost_list.append(cost(X,Y,Theta))

    return Theta,cost_list

iteraion = 100

alphe = 0.1

j,cost_lst=GD(x,y,theta,alphe,iteraion)

print("New Theta = ",j)

print("New COST = ",cost(x,y,j))

## the predict price for feet

hox = j[0,0]+ j[0,1] * data.feet.values

plt.scatter(data.feet.values,data.price.values,color="b",label="Apartment With Feet & Real Price")

plt.plot(data.feet.values,hox,color="r",label="The Predict Price For Apartment")

plt.legend()

plt.xlabel("The Feet")

plt.ylabel("The Price")

plt.show()

### draw the cost value in GD

xcost= []

for i in range(iteraion):

      xcost.append(i)

plt.plot(xcost,cost_lst)

plt.show()