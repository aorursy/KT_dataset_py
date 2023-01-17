###############################################################
#       Step 1 : Importing the libraries                      #
###############################################################


# NumPy is module for Python. The name is an acronym for "Numeric Python" or "Numerical Python".
# This makes sure that the precompiled mathematical and numerical functions 
# and functionalities of Numpy guarantee great execution speed.

import numpy as np

# Pandas is an open-source Python Library providing high-performance data manipulation 
# and analysis tool using its powerful data structures. 
# The name Pandas is derived from the word Panel Data – an Econometrics from Multidimensional data.

import pandas as pd


# The OS module in Python provides a way of using operating system dependent functionality. 
# The functions that the OS module provides allows you to interface with the underlying operating system 
# that Python is running on – be that Windows, Mac or Linux.

import os
###############################################################
#       Step 2 : Importing the Dataset                        #
###############################################################

#Read the 'Data.csv' and store the data in the vairable dataset.
dataset = pd.read_csv("../input/Data.csv")
print('Load the datasets...')


# Print the shape of the dataset
print ('dataset: %s'%(str(dataset.shape)))

# print the dataset
dataset
# Separate the dependent and independent variables

# Independent variable
# iloc[rows,columns]
# Take all rows
# Take last but one column from the dataset (:-1)
X = dataset.iloc[:,:-1].values

# Dependent variable
# iloc[rows,columns]
# Take all rows
# Take last column from the dataset (:-1)
Y = dataset.iloc[:,3].values
# Print the X and Y
print ('X: %s'%(str(X)))
print ('-----------------------------------')
print ('Y: %s'%(str(Y)))
###############################################################
#       Step 3 : Missing Data                                 #
###############################################################

# Scikit-learn provides a range of supervised and unsupervised learning algorithms via a consistent interface in Python.
# The sklearn.preprocessing package provides several common utility functions and transformer classes 
# to change raw feature vectors into a representation that is more suitable for the downstream estimators.

from sklearn.preprocessing import Imputer

# Imputer Class takes the follwing parameters:
#     missing_values : The missing values in our dataset are called as NaN (Not a number).Default is NaN
#     strategy       : replace the missing values by mean/median/mode. Default is mean.
#     axis           : if axis = 0, we take we of the column and if axis = 1, we take mean value of row.

imputer = Imputer(missing_values = 'NaN',strategy = 'mean', axis = 0)

# Fit the imputer on X.
# Take all rows and columns only with the missing values.
# Note: Index starts with 0. Upper bound (3) is not included.

# Fit imputer for columns 1 and 2 of X matrix.
imputer = imputer.fit(X[:,1:3])

#Replace missing data with mean of column
X[:,1:3] = imputer.transform(X[:,1:3])

print ('X: %s'%(str(X)))
###############################################################
#       Step 4 : Categorical variables                        #
###############################################################

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X[:,0]
# Applying the OneHotEncoder to the first column[0]
onhotencoder = OneHotEncoder(categorical_features = [0])
X=onhotencoder.fit_transform(X).toarray()

# Encoding the categorical data for Y matrix
labelencoder_Y = LabelEncoder()
Y = labelencoder_X.fit_transform(Y)
Y
###############################################################
#       Step 5 : Splitting the dataset                        #
###############################################################

from sklearn.cross_validation import train_test_split

# The test size is taken as 20% of the total dataset i.e., out of 15 only 3 rows are taken as test set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)
# Print the shape of the dataset
print ('X_train: %s'%(str(X_train.shape)))
print ('----------------')
print ('X_test: %s'%(str(X_test.shape)))
print ('----------------')
print ('Y_train: %s'%(str(Y_train.shape)))
print ('----------------')
print ('Y_test: %s'%(str(Y_test.shape)))
print ('----------------')
###############################################################
#       Step 6 : Feature Scaling                              #
###############################################################

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

# We need to fit and transform the training set
X_train = sc_X.fit_transform(X_train)

# We need to fit the test set
X_test = sc_X.transform(X_test)
X_train
X_test