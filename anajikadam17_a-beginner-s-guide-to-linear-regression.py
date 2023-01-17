import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv("../input/salary_data.csv")

X = dataset.iloc[:, :-1].values #get a copy of dataset exclude last column

y = dataset.iloc[:, 1].values #get array of dataset in column 1st
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
# Visualizing the Training set results



plt.scatter(X_train, y_train, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title('Salary VS Experience (Training set)')

plt.xlabel('Year of Experience')

plt.ylabel('Salary')

plt.show()



# Visualizing the Test set results



plt.scatter(X_test, y_test, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title('Salary VS Experience (Test set)')

plt.xlabel('Year of Experience')

plt.ylabel('Salary')

plt.show()
# Predicting the result of 5 Years Experience



y_pred = regressor.predict([[5.0]])

y_pred
# Predicting the Test set results

# type(X_test)

y_pred = regressor.predict(X_test)

y_pred
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv("../input/salary_data.csv")

X = dataset.iloc[:,1:].values #get a copy of dataset exclude last column

y = dataset.iloc[:, 0].values #get array of dataset in column 1st



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)



# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)



# Visualizing the Training set results



plt.scatter(X_train, y_train, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title('Salary VS Experience (Training set)')

plt.ylabel('Year of Experience')

plt.xlabel('Salary')

plt.show()



# Visualizing the Test set results



plt.scatter(X_test, y_test, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title('Salary VS Experience (Test set)')

plt.ylabel('Year of Experience')

plt.xlabel('Salary')

plt.show()


y_pred = regressor.predict(X_test)

y_pred
# Predicting the result for 5 Years Experience by given privious result 



y_pred = regressor.predict([[73545.90445964]])

y_pred