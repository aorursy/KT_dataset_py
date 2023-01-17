#Importing the neccessary modules for data manipulation and visual representation.

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline



#Importing the Dataset.

dataset = pd.read_csv('../input/Salary_Data.csv')

x = dataset.iloc[:,:-1].values

y = dataset.iloc[:,1].values
#splitting the Dataset into Training set and Test set.

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 1/3,random_state = 0)
#Fitting Simple linear regression to the training set.

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)
#Predicting the test set result.

y_pred = regressor.predict(x_test)
#visualising the Training set result.

plt.scatter(x_train,y_train,color ='green')

plt.plot(x_train,regressor.predict(x_train),color = 'blue')

plt.title('salary vs experience(training set)')

plt.xlabel('years of experience')

plt.ylabel('salary')

plt.show()
#visualising the Test set result.

plt.scatter(x_test,y_test,color ='green')

plt.plot(x_train,regressor.predict(x_train),color = 'blue')

plt.title('salary vs experience(test set)')

plt.xlabel('years of experience')

plt.ylabel('salary')

plt.show()