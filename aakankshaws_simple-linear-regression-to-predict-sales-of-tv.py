#import libraries

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings

import os

print(os.listdir("../input"))

#import and read data set

data = pd.read_csv('../input//tvmarketing.csv')

data.head(5)
#look for more information about the data set

data.info()
#look up the statistics of the numeric data 

data.describe()
#check the shape of data

data.shape
sns.pairplot(data, x_vars='TV', y_vars='Sales', height=5,  kind='scatter' )
# Putting feature variable to X

X = data['TV']

X.head()
#Putting response variavle to y

y = data['Sales']

y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)
print(type(X_train))

print(type(X_test))

print(type(y_train))

print(type(y_test))
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
#It is a general convention in scikit-learn that observations are rows, while features are columns. 

#This is needed only when you are using a single feature; in this case, 'TV'.

import numpy as np



X_train = X_train[:, np.newaxis]

X_test = X_test[:, np.newaxis]
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression



# Representing LinearRegression as lr(Creating LinearRegression Object)

lr = LinearRegression()



# Fit the model using lr.fit()

lr.fit(X_train, y_train)
# Print the intercept and coefficients

print(lr.intercept_)

print(lr.coef_)
y_pred = lr.predict(X_test)
#check actual and predicted values

c = [number for number in range(1,61,1)] #generate index

fig = plt.figure()

plt.plot(c, y_test, label='actual')

plt.plot(c, y_pred, label='predicted')

plt.title('Actual vs. Predicted values')

plt.xlabel('index')

plt.ylabel('Sales')

plt.legend()
fig = plt.figure()

plt.plot(c, y_test-y_pred, label='actual')

plt.title('Error values')

plt.xlabel('index')

plt.ylabel('y_test-y_pred')

plt.legend()
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
import matplotlib.pyplot as plt

plt.scatter(y_test,y_pred)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')