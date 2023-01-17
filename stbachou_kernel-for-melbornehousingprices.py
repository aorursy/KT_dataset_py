# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Use code-checking and import libs



import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex2 import *

print("Setup Complete")

#Data initialization



# save filepath to variable for easier access

train = pd.read_csv('../input/traintestandsample/train.csv')

test = pd.read_csv('../input/traintestandsample/test.csv')

melbourne_file_path = '../input/melbournehousingprices/melb_data.csv'

# read the data and store data in DataFrame titled melbourne_data

melbourne_data = pd.read_csv(melbourne_file_path) 



#drop shit we dont need

train = train.drop(['MSZoning', 'Street', 'Alley'],axis=1)

test = test.drop(['MSZoning', 'Street', 'Alley'],axis=1)

melbourne_data = melbourne_data.drop(['Type', 'Address', 'SellerG'],axis=1)

# Call line below with no argument to check that you've loaded the data correctly

step_1.check()

# Just some random experiments with solution and hint function

step_1.hint()

step_1.solution() #why does the solution function refer to the iowa project?

# print a summary of the data in Melbourne data

melbourne_data.describe()

#heads of test and train data

train.head(1000)
#Data review



max_line = melbourne_data.max()

print(max_line)



# What is the average lot size (rounded to nearest integer)?

# todo



# As of today, how old is the newest home (current year - the date in which it was built)

newest_home_age = melbourne_data.YearBuilt.max()

print(newest_home_age)
#Todo: Proceed with chapter "Your First Machine Learning Model", then start working towards a solution

#https://www.kaggle.com/dansbecker/your-first-machine-learning-model





#prediction target and features playground



#list all columns in the dataset

melbourne_data_unimputed = pd.read_csv(melbourne_file_path) 

#melbourne_data['Id'] = melbourne_data.index.values



# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)

# We'll learn to handle missing values in a later tutorial.  

# Your Iowa data doesn't have missing values in the columns you use. 

# So we will take the simplest option for now, and drop houses from our data. 

# Don't worry about this much for now, though the code is:





# dropna drops missing values (think of na as "not available")

#from sklearn.impute import SimpleImputer

#my_imputer = SimpleImputer()

#melbourne_data = my_imputer.fit_transform(melbourne_data_unimputed)

melbourne_data = melbourne_data.dropna(axis=0)

train = train.drop([1459])

train = train.fillna('0')

test = test.fillna('0')

#after

melbourne_data.describe()





#Two way to select a subset

#

#  -Dot notation, which we use to select the "prediction target" (y) -> column, we want to predit

#  -Selecting with a column list, which we use to select the "features" (x)

#



y = train.SalePrice



melbourne_features = ['YearBuilt', 'OverallQual', 'LotArea', 'GarageArea', 'LotFrontage']



# store features

X = train[melbourne_features]



#show head out of x (first five)

X.head()









#Using scikit-learn



#Steps



#    Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.

#    Fit: Capture patterns from provided data. This is the heart of modeling.

#    Predict: Just what it sounds like

#    Evaluate: Determine how accurate the model's predictions are.



# Define model. Specify a number for random_state to ensure same results each run

melbourne_model = DecisionTreeRegressor(random_state=1)



# Fit model (X features, y prediction target)

melbourne_model.fit(X, y)



print("Making predictions for the following 5 houses:")

print(X.head())

print("The predictions are")



#predict probably checks predicition target and features and predicts future 

print(melbourne_model.predict(X.head()))
#Create csv





#melbourne_features.

#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'Id': test['Id']

                           ,

                           'SalePrice':melbourne_model.predict(X)})





#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'MelbourneHousingPricesSubmition2.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)