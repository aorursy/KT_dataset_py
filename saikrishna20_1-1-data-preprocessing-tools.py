import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
dataset = pd.read_csv('../input/data-preprocessing/Data.csv')
# x[lowerbound : upperbound] gives a range btwn lower and upperbound
# iloc it means the index location so we have to provide the indexes i.e int
# : colon means range, when used in iloc[:,:] which means that all the elements
# if there are no numbers before or after colon which by default takes the lowerbound
# and the upperbound which means all the rows and all the columns.
X = dataset.iloc[:, :-1].values
# .values convert dataframe to np array as the time taken for 
# array calculation are less so we convert it to array
# in the above step we have given iloc[:,:-1] meaning there is no defined upper 
# and lower bound so it takes all the rows, but for the columns it's given
# :-1 there is no number before colon so it takes the default lowerbound i.e from the starting
# and after colon the upperbound is defined as -1 meaning the last column or first column from the last
# as we know in python the right index is excluded so all the columns untill 
# the first col from last will be selected
y = dataset.iloc[:, -1].values
# in this code all the rows and only the last column is selected
# we divided the dataset into X and y bcoz X are independent and y is dependent variable
dataset
print(X)
#all independent variables
print(y)
#all independent outcomes
from sklearn.impute import SimpleImputer
# sklearn also known as sci kit learn is the free machine learning library in python
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
# Simple Imputer (class) used to fill up the missing values with a defined 
# strategy(mean, median, most_frequency, constant)
# creating an object of Simple Imputer with strategy of Mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# fitting the imputer so it calculates the mean of the specified columns
imputer.fit(X[:, 1:3])
# transform is when the mean values are placed in the dataset
X[:, 1:3] = imputer.transform(X[:, 1:3])
# we want to specify the only columns we need to fill the values
print(X)
from sklearn.compose import ColumnTransformer
# Column Transformer is a class and it Applies transformers to columns of an array or pandas DataFrame.
from sklearn.preprocessing import OneHotEncoder
# OneHotEncoder is a class object used to encode categeorical
# Encode categorical features as a one-hot numeric array
# converts categorical to dummy variables so the program can understand
# create and object and pass the functions
# https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html?highlight=column%20transformer#sklearn.compose.ColumnTransformer
# List of (name, transformer, columns)
# name is encoder, tranformer is a OneHotEncoder function, column is [0] means 1st
# ‘first’ : drop the first category in each feature. If only one category is 
# present, the feature will be dropped entirely.
# remainder = 'passthrough' means all the col which are not specified
#  will be passed through
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop= 'first'), [0])], remainder='passthrough')
# now we fit here and now the data of col gets converted into dummy variables
X = np.array(ct.fit_transform(X))
print(X)
from sklearn.preprocessing import LabelEncoder
# LabelEncoder is a class used to encode Yes/No
# Encode target labels with value between 0 and n_classes-1.
#This transformer should be used to encode target values, i.e. y, and not the input X.
le = LabelEncoder() # object
y = le.fit_transform(y)# fit and here they get converted to 0,1
print(y)
from sklearn.model_selection import train_test_split
# for every supervised model we do do the split so that there is no over fitting
# means the model works well on training data but not testing data
# to avoid that we use split and we give test_size 0.2 ie 20 % of overall data
# to reproduce the same result we give some random_state 
# train_test_split takes up two variables
# X_train, X_test, y_train, y_test it is called unpacking, geting all values at once
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
# there are 2 popular feature scaling techniques they are standardisation and 
# Normalisation, we prefer standardisation as it's output values will be in the
#range of -3 to +3 and can be applied to any kind of distributed Data.
# Normalisation works well with normally distributed Data and its range 0 to 1
# Standardisation works with every data so it's preferred.
# not all models require scaling the data 
# we don't scale the dummy variables because it has no use in the change of accuracy
from sklearn.preprocessing import StandardScaler # class
sc = StandardScaler() # object
X_train[:, 2:] = sc.fit_transform(X_train[:, 2:])# here all the mean and Std(standard deviation) is calculated
X_test[:, 2:] = sc.transform(X_test[:, 2:]) # the values are standardised in this step.
print(X_train)
print(X_test)