

#Importing libraries.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os
#Read the datasets

dataset=pd.read_csv("../input/weight-height.csv")

#Check the dataset

dataset.head()





#check if null values is present in dataset or not.....

dataset.isnull().sum()

#Checking the dimensions of the dataset

dataset.shape


#Plot Gender vs Weight............

x1 = dataset.iloc[:, 0].values

y1 = dataset.iloc[:, 2].values

plt.scatter(x1,y1,label='Gender',color='Green',s=50)

plt.xlabel('Gender')

plt.ylabel('Weight')

plt.title('Gender vs Weight')

plt.legend()

plt.show()



#Plot Height vs Weight............

x2 = dataset.iloc[:, 1].values

y2 = dataset.iloc[:, 2].values

plt.scatter(x2,y2,label='Height',color='Orange',s=50)

plt.xlabel('Height')

plt.ylabel('Weight')

plt.title('Height vs Weight')

plt.legend(loc="lower right")

plt.show()

#separating the dependent and independent values....

#X-Independent variable

X = dataset.iloc[:, 1:2].values

print(X)

#y-dependent or target variable

y = dataset.iloc[:, 2].values

print(y)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#Creating linear regression model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



#predicting the test set

y_pred = regressor.predict(X_test)



# Visualising the Training set results

plt.scatter(X_train, y_train, color = 'Yellow')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Hight vs Weights (Training set)')

plt.xlabel('Hight')

plt.ylabel('Weight')

plt.show()



# Visualising the Test set results

plt.scatter(X_test, y_test, color = 'Yellow')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Hight vs weights (Test set)')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.show()
# Predicting the Test set results

y_pred = regressor.predict(X_test)

print('Coefficients: ', regressor.coef_)

# The mean squared error

print("Mean squared error: %.2f" % np.mean((regressor.predict(X_test) - y_test) ** 2))

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % regressor.score(X_test, y_test))






