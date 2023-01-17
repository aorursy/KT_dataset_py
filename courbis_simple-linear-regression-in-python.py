# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('/kaggle/input/heights-and-weights/data.csv')
dataset.head()
# creating independent variables
X = dataset.iloc[: ,:-1].values
X
# creating dependent variables
y = dataset.iloc[:, 1].values
y
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

X_train
y_test
#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#Predicting the test set results
y_pred = regressor.predict(X_test)

# Visualizing the Training set results
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Weight vs Height (Training set)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()
# Visualizing the Test set results
plt.scatter(X_test,y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Weight vs Height (Test set)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()