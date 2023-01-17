import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

print(os.listdir("../input"))



Aus_Hous_Data1=pd.read_csv("../input/data.csv")

Aus_Hous_Data1.tail()
# Checking for null values

Aus_Hous_Data1.isnull().sum()
Aus_Hous_Data2=Aus_Hous_Data1[['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement']]

Aus_Hous_Data2.head()
#Applying standardization

Aus_Hous_Data2=(Aus_Hous_Data2-Aus_Hous_Data2.mean())/Aus_Hous_Data2.std()

x=Aus_Hous_Data2[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','view','condition','sqft_above','sqft_basement']]

y=Aus_Hous_Data2['price']

x.shape,y.shape                 

                  
x = np.c_[np.ones(x.shape[0]), x]
alpha = 0.01 #Step size

iterations = 3000 #No. of iterations

np.random.seed(123) #Set the seed

theta = np.random.rand(10) #Pick some random values to start with
# Stochastic gradient descent (SGD) updates the parameters for each training example within the dataset, one by one.

def stochastic_gradient_descent(x,y,theta,b,alpha,iteration):

    past_costs = []

    for i in range(iteration):

        for row,target in zip(x,y):

            row=row.reshape(1,10)

            m=row.shape[0]

            y_pred=row.dot(theta)+b

            loss=y_pred-target

            #cost = 1/(2*m) * np.dot(loss.T, loss)

            #past_costs.append(cost)

            theta=theta-(2/float(m))*alpha*(row.T.dot(loss))

            b=b-(2/float(m))*alpha*sum(loss)

           # print(past_costs)

    MSE=mean_squared_error(y, x.dot(theta)+b)         

    return MSE
MSE = stochastic_gradient_descent(x, y, theta, 1, alpha,iterations)

#print("Cost:",Cost)
print("Mean square error :",MSE)