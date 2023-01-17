# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results written to the current directory are saved as output.
# load training data
df_train = pd.read_csv('../input/traindata.csv') # import training data from CSV

# load test data
df_test = pd.read_csv('../input/testdata.csv')
# Get the basic information of the data using the pandas package
# (only the result from the last executed command will be shown)

# view first 10 rows of the dataframe
#df_train.head(10)
#df_test.head(10)

# get descriptive statistics about the data in the dataframe
#df_train.describe()
#df_test.describe()

# get the datatypes of the different columns in the dataframe
df_train.dtypes
#df_test.dtypes
# convert dataframe to array by indexing it ('id' column used as index)
df_train.set_index("id", inplace=True)

# copy the target variables (here the 'purpose_validated' column into a new array
# --> this returns the training target list
y_train = df_train['purpose_validated'].values

# copy all columns except the 'purpose_predicted' of the training dataframe into a new array
# --> this returns the training features
X_train = df_train.drop(['purpose_validated'],axis=1).values
# convert dataframe to array by indexing it ('id' column used as index)
df_test.set_index("id", inplace=True) # use id as the index for the data

# copy all columns of the test dataframe into a new array
X_test = df_test.values
# print created lists for a quick validation

#print(y_train)
#print(X_train)
print(X_test)
# import the label binarizer
from sklearn.preprocessing import LabelBinarizer

# create LabelBinarizer object
lb = LabelBinarizer()

# 'tell' the binarizer which categories exist by feeding it a set of data
lb.fit(y_train)

# transform the target variable, i.e. apply the one-hotting
y_train = lb.transform(y_train)
# print the created one-hotting or the found classes for a quick validation

lb.classes_
#print(y_train)
# import standardisation object
from sklearn.preprocessing import StandardScaler

# create StandardScaler object
scaler = StandardScaler()

# fit the training data to the scaler
scaler.fit(X_train)

# transform the training and testing data with the scaler (features only)
scaler.transform(X_train)
scaler.transform(X_test)
# import the random forest regressor
from sklearn.ensemble import RandomForestRegressor

# create random forest regression object
# set number of estimators to 100
rf = RandomForestRegressor(100)

# train the model by fitting the training features and target lists
rf.fit(X_train, y_train)

# predict the results for the test features
y_predicted_rf = rf.predict(X_test)
# transform predicted values back into categorical data, i.e. undo one-hotting
y_predicted_rf = lb.inverse_transform(y_predicted_rf)

# print the results for a quick validation
print(y_predicted_rf)
# copy test features to a new dataframe
df_submission = df_test.copy()

# add the predicted values for the purpose as a new column
df_submission['purpose_validated'] = y_predicted_rf

# 
df_submission = df_submission['purpose_validated']

# convert dataframe to CSV-file (with header and 'id' as the index)
df_submission.to_csv('result_logisticReg.csv', header=True, index='id')