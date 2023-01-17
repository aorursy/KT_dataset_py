"""

@author: shyam

"""

#Importing Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#Importing Datasets

""""

In the following data

X = number of claims

Y = total payment for all the claims in thousands of Swedish Kronor for geographical zones in Sweden

"""

data = pd.read_excel('../input/Insurance.xls')

X = data.iloc[:,:-1]

y = data.iloc[:,1]
#Splitting the dataset

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3)
#Fitting the model to our training data

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
#Predicting the values

y_pred = regressor.predict(X_test)
####Visualizing the training set results

plt.scatter(X_train, y_train , color = 'red')

plt.plot(X_train,regressor.predict(X_train) , color = 'blue')

plt.title('Claims Vs Payment (Test Set)')

plt.xlabel('Claimns')

plt.ylabel('Payment')

plt.show()
####Visualizing the test set results

plt.scatter(X_test, y_test , color = 'red')

plt.plot(X_train,regressor.predict(X_train) , color = 'blue')

plt.title('Claims Vs Payment (Test Set)')

plt.xlabel('Claimns')

plt.ylabel('Payment')

plt.show()
####Variance score of the model

from sklearn.metrics import explained_variance_score

vs = explained_variance_score(y_test,y_pred)

print(vs)