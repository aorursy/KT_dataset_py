import pandas as pd                  # A fundamental package for linear algebra and multidimensional arrays

import numpy as np                   # Data analysis and data manipulating tool

import random                        # Library to generate random numbers

from collections import Counter      # Collection is a Python module that implements specialized container datatypes providing 

                                     # alternatives to Python’s general purpose built-in containers, dict, list, set, and tuple.

                                     # Counter is a dict subclass for counting hashable objects

# Visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



# To ignore warnings in the notebook

import warnings

warnings.filterwarnings("ignore")
# This is a subset of the original data available at kaggle.

fraud_data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Imbalanced_classes/master/fraud_data.csv")
fraud_data.head()
fraud_data.info()      # Returns a concise summary of dataset
# Taking a look at the target variable

fraud_data.isFraud.value_counts()       # The value_counts() function is used to get a Series containing counts of unique values.
fraud_data.isFraud.value_counts() / len(fraud_data) * 100       # Gets the percentage of unique values in the variable 'isFraud'
# we can also use countplot form seaborn to plot the above information graphically.

sns.countplot(fraud_data.isFraud)
fraud_data.isnull().sum() / len(fraud_data) * 100   # To get percentage of missing data in each column
fraud_data = fraud_data[fraud_data.columns[fraud_data.isnull().mean() < 0.2]]    # Will keep those columns which has missing values less than 20%
# filling missing values of numerical columns with mean value.

num_cols = fraud_data.select_dtypes(include=np.number).columns      # getting all the numerical columns



fraud_data[num_cols] = fraud_data[num_cols].fillna(fraud_data[num_cols].mean())   # fills the missing values with mean
cat_cols = fraud_data.select_dtypes(include = 'object').columns    # getting all the categorical columns



fraud_data[cat_cols] = fraud_data[cat_cols].fillna(fraud_data[cat_cols].mode().iloc[0])  # fills the missing values with maximum occuring element in the column



"""

Explaining above line:



The above line of code is replacing the missing values in the columns in cat_cols with the mode (most repeated elements) of the non-missing values 

in the same columns.

The .iloc[0] attribute is selecting just the first mode returned, in case they are multiple values with the same highest frequency of occurrence. 

Please review the documentation for further clarifications on this regard: 

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mode.html

"""
# Let's have a look if there still exist any missing values

fraud_data.isnull().sum() / len(fraud_data) * 100
fraud_data = pd.get_dummies(fraud_data, columns=cat_cols)    # earlier we have collected all the categorical columns in cat_cols
fraud_data.head()
# Separate input features and output feature

X = fraud_data.drop(columns = ['isFraud'])       # input features

Y = fraud_data.isFraud      # output feature
from sklearn.preprocessing import StandardScaler

scaled_features = StandardScaler().fit_transform(X)

scaled_features = pd.DataFrame(data=scaled_features)

scaled_features.columns= X.columns
# Let's see how the data looks after scaling

scaled_features.head()
# Splitting the data

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)



# X_train: independent feature data for training the model

# Y_train: dependent feature data for training the model

# X_test: independent feature data for testing the model; will be used to predict the target values

# Y_test: original target values of X_test; We will compare this values with our predicted values.

 

# test_size = 0.3: 30% of the data will go for test set and 70% of the data will go for train set

# random_state = 42: this will fix the split i.e. there will be same split for each time you run the code
# 'resample' is located under sklearn.utils

from sklearn.utils import resample
# concatenate training data back together

train_data = pd.concat([X_train, Y_train], axis = 1)
# separate minority and majority class

not_fraud = train_data[train_data.isFraud==0]

fraud = train_data[train_data.isFraud==1]
# Unsample minority; we are oversampling the minority class to match the number of majority classs

fraud_upsampled = resample(fraud,

                           replace = True, # Sample with replacement

                           n_samples = len(not_fraud), # Match number in majority class

                           random_state=27)
# combine majority and upsampled minority

upsampled = pd.concat([not_fraud, fraud_upsampled])
# Now let's check the classes count

upsampled.isFraud.value_counts()
# we are still using our separated class i.e. fraud and not_fraud from above

# Again we are removing the observations of the majority class to mathch the number of minority class

# downsample majority

not_fraud_downsampled = resample(not_fraud,

                                replace = False, # sample without replacement

                                n_samples = len(fraud), # match minority n

                                random_state = 27)
# combine minority and downsampled majority

downsampled = pd.concat([not_fraud_downsampled, fraud])    # Concatenation
# let's check the classes counts

downsampled.isFraud.value_counts()
# import SMOTE 

from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state = 25, sampling_strategy = 1.0)   # again we are eqalizing both the classes
# fit the sampling

X_train, Y_train = sm.fit_sample(X_train, Y_train)
np.unique(Y_train, return_counts=True)     # Y_train is numpy array, so unique() functions returns the count of all the unique elements in the array