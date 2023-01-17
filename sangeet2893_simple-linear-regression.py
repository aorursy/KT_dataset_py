# Importing required libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline
# Loading data

data = pd.read_csv('../input/advertising.csv/Advertising.csv')
# First few rows of data

data.head()
plt.figure(figsize=(14,4))



plt.subplot(1,3,1)

plt.scatter(data['TV'], data['sales'], color='blue')

plt.xlabel('TV')

plt.ylabel('sales')



plt.subplot(1,3,2)

plt.scatter(data['radio'], data['sales'], color = 'red')

plt.xlabel('radio')

plt.ylabel('sales')



plt.subplot(1,3,3)

plt.scatter(data['newspaper'], data['sales'], color = 'green')

plt.xlabel('newspaper')

plt.ylabel('sales')



plt.show()
# Defining X & Y



X = data['TV'].values.reshape(-1,1)

Y = data['sales'].values.reshape(-1,1)
# Splitting data into Train and Test datasets

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Fitting the Linear Regression Model

Model = LinearRegression()

Model.fit(X_train, y_train)
train_fit = Model.predict(X_train)

test_pred = Model.predict(X_test) 
plt.figure(figsize=(12,4))



plt.subplot(1,2,1)

plt.scatter(X_train, y_train, color='gray')

plt.plot(X_train, train_fit, color='blue', linewidth=2)

plt.xlabel('TV')

plt.ylabel('sales')

plt.title("Train Dataset")



plt.subplot(1,2,2)

plt.scatter(X_test, y_test, color='gray')

plt.plot(X_test, test_pred, color='blue', linewidth=2)

plt.xlabel('TV')

plt.ylabel('sales')

plt.title("Test Dataset")



plt.show()
#To print the value of Intercept:

print(Model.intercept_)



# To print the value of x-coefficient or slope:

print(Model.coef_)
plt.figure(figsize=(10,7))

plt.scatter(X_train, y_train,  color='firebrick', s=20)

plt.plot(X_train, train_fit, color='royalblue', linewidth=0.7)

for i in range(X_train.shape[0]):

    plt.plot([X_train[i], X_train[i]], [y_train[i], train_fit[i]], color='black', linewidth=0.5)

plt.xlabel("TV")

plt.ylabel("sales")

plt.title("Residuals in the data after fitting the model")

plt.text(240,3,"@DataScienceWithSan")

plt.text(0, 23, "Verical lines represent\nthe distance between\neach data point and \nthe regression line of model")

plt.show()
print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_pred))  