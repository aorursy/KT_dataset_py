import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from sklearn import metrics

df = pd.read_csv("../input/trialdata/trial.csv")

X = df.iloc[:,:-1].values

y = df.iloc[:,1].values



X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = .3, random_state=0)



regressor = LinearRegression()

regressor.fit(X_train, y_train)

print(regressor.coef_)

print(regressor.intercept_)



y_pred = regressor.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print(regressor.score(X_test,y_test))



plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = "blue")

plt.xlabel("Exp")

plt.ylabel("Sal")

plt.show()



plt.scatter(X_test, y_test, color="red")

plt.plot(X_test, y_pred, color="blue")

plt.xlabel("Exp")

plt.ylabel("Sal")

plt.show()

# Code you have previously used to load data

import pandas as pd



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



home_data = pd.read_csv(iowa_file_path)



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex3 import *



print("Setup Complete")
# print the list of columns in the dataset to find the name of the prediction target

y = ____



# Check your answer

step_1.check()
# The lines below will show you a hint or the solution.

# step_1.hint() 

# step_1.solution()
# Create the list of features below

feature_names = ___



# Select data corresponding to features in feature_names

X = ____



# Check your answer

step_2.check()
# step_2.hint()

# step_2.solution()
# Review data

# print description or statistics from X

#print(_)



# print the top few lines

#print(_)
# from _ import _

#specify the model. 

#For model reproducibility, set a numeric value for random_state when specifying the model

iowa_model = ____



# Fit the model

____



# Check your answer

step_3.check()
# step_3.hint()

# step_3.solution()
predictions = ____

print(predictions)



# Check your answer

step_4.check()
# step_4.hint()

# step_4.solution()
# You can write code in this cell
