# Code you have previously used to load data

import pandas as pd



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



home_data = pd.read_csv(iowa_file_path)



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex3 import *



home_data.head()
# print the list of columns in the dataset to find the name of the prediction target



y = home_data.SalePrice



print(y)

# Create the list of features below

feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']



# Select data corresponding to features in feature_names

X = home_data[feature_names]



# Check your answer

X
# print the top few lines

X.head()
# Review data

# print description or statistics from X



X.describe()
from sklearn.tree import DecisionTreeRegressor

#specify the model. 



#For model reproducibility, set a numeric value for random_state when specifying the model

model = DecisionTreeRegressor(random_state=1)









# Fit the model

model.fit(X, y)
predictions = ____

print(predictions)



# Check your answer

step_4.check()
print("Making predictions for the following 5 houses:")

print(X.head())

print("The predictions are")

print(model.predict(X.head()))