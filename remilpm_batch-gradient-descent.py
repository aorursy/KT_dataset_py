import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.utils import shuffle

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

import copy

from sklearn.metrics import mean_squared_error

print(os.listdir("../input"))



House1= pd.read_csv("../input/data.csv")

House1.tail()
import copy

House2= copy.deepcopy(House1)

House2.head()
House2=House2[['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above',

               'sqft_basement','yr_built','yr_renovated']]

House3=(House2-House2.mean())/House2.std()

House3.head()

House3=(House3-House3.mean())/House3.std()

x=House3[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated']]

y=House3['price']   

x.shape,y.shape
x = np.c_[np.ones(x.shape[0]), x]
#MINI BATCH GRADIENT DESCENT



alpha = 0.01 #Step size

iterations = 3000 #No. of iterations

np.random.seed(123) #Set the seed

theta = np.random.rand(13) #Pick some random values to start with
#MINI BATCH GRADIENT DESCENT

def Mini_Batch_gradient_descent(x, y, theta, iteration, alpha,batch):

    m=x.shape[0]

    b=0

    prediction_list=[]

    for i in range(iteration):

        idx = np.random.randint(m, size=batch)

        x1=x[idx,:]

        y1=y[idx]

        m1=x1.shape[0]

        y_pred=x1.dot(theta)+b

        prediction_list.append(y_pred)

        loss=y_pred-y1

        cost=(1/2*float(m1))*np.sum(np.square(loss))

        theta=theta-(2/float(m1))*alpha*(x1.T.dot(loss))

        b=b-(2/float(m1))*alpha*sum(loss)

    MSE=mean_squared_error(y, x.dot(theta)+b)   

    return theta,cost,prediction_list,MSE

        

   





theta,cost,prediction_list,MSE = Mini_Batch_gradient_descent(x, y, theta, iterations, alpha,100)





print("Root Mean square error:",np.sqrt(MSE))