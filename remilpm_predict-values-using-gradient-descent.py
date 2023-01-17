import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from sklearn import preprocessing

print(os.listdir("../input"))

Housing_Data1=pd.read_csv("../input/output.csv")

Housing_Data1.head()
Housing_Data1.shape
#Checking for null values

Housing_Data1.isnull().sum()
import copy

Housing_Data2=copy.deepcopy(Housing_Data1)

Housing_Data2.head()
#Apply standardization

Housing_Data3=Housing_Data2[['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated']]

Housing_Data3=(Housing_Data3-Housing_Data3.mean())/Housing_Data3.std()

Housing_Data3.head()
#Split data in to training and test dataset





X = Housing_Data3.as_matrix(['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated'])

y = Housing_Data3['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=10)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train= sc.transform(X_train)

X_test = sc.transform(X_test)

       #lets scale the data

X = Housing_Data3.as_matrix(['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated'])

y = Housing_Data3['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=10)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train= sc.transform(X_train)

X_test = sc.transform(X_test)

X_test.shape,X_train.shape,y_train.shape,y_test.shape
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

#let us predict

y_pred=model.predict(X_test)

print (model.score(X_test, y_test))
y_pred.shape
#Mean Square error of the prediction

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Root mean squared error is {rmse}")

#Initiating Gradient Descent calculation

Housing_Data4=Housing_Data2[['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated']]

Housing_Data4.head()
Housing_Data4=(Housing_Data4-Housing_Data4.mean())/Housing_Data4.std()

x=Housing_Data4[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated']]

y=Housing_Data4['price']    







                 
x = np.c_[np.ones(x.shape[0]), x] 

#GRADIENT DESCENT



alpha = 0.01 #Step size

iterations = 12000 #No. of iterations

m = y.size #No. of data points

#y.size

np.random.seed(123) #Set the seed

theta = np.random.rand(13) #Pick some random values to start with
#GRADIENT DESCENT

def gradient_descent(x, y, theta, iterations, alpha):

    past_costs = []

    past_thetas = [theta]

    prediction_list=[]

    for i in range(iterations):

        prediction = np.dot(x, theta)

        prediction_list.append(prediction)

        error = prediction - y

        cost = 1/(2*m) * np.dot(error.T, error)

        past_costs.append(cost)

        theta = theta - (alpha * (1/m) * np.dot(x.T, error))

        #print(theta)

        past_thetas.append(theta)

        

    return prediction_list,past_thetas, past_costs,error



#Pass the relevant variables to the function and get the new values back...

prediction_list,past_thetas, past_costs,error = gradient_descent(x, y, theta, iterations, alpha)

theta = past_thetas[-1]



#Print the results...

#print("error:",error)



#print("prediction_list:",prediction_list)
MSE_GD = ((prediction_list[-1]-y)**2).mean()  #From Gradient Descent

RMSE_GD=np.sqrt(MSE_GD)

print('Root Mean Square Error from Gradient Descent prediction : {}'.format(round(RMSE_GD,3)))

#Plot the cost function...

plt.title('Cost Function J')

plt.xlabel('No. of iterations')

plt.ylabel('Cost')

plt.plot(past_costs)

plt.show()