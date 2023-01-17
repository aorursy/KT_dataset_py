# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Plotting
import seaborn as sns
# Scaling preprocessing library
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing
from sklearn.preprocessing import Imputer
# Math Library
from math import ceil
# Boosting Libraries
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
# Keras importing
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read the train data 
train = pd.read_csv('../input/yds_train2018.csv')
# print the top 5 row from the dataframe
train.head()
# read the test data
test = pd.read_csv('../input/yds_test2018.csv')
test.head()
# Matching the train and test columns
[c for c in train.columns if c not in test.columns]
# Looking for NaN values 
droping_list_all=[]
for j in range(0,8):
    if not train.iloc[:, j].notnull().all():
        droping_list_all.append(j)        
droping_list_all
# Conditional groupby of train data

df_grouped = pd.DataFrame(train.groupby(['Year','Month','Product_ID','Country'])['Sales'].sum().reset_index())
# Sorting the grouped dataframes index
df_grouped = df_grouped.sort_index()
# Look for top 5 rows
df_grouped.head()

# Look for last 5 rows
df_grouped.tail()
# Importing the expense data for simulation
df_expense = pd.read_csv('../input/promotional_expense.csv')
df_expense.head()
# Plotting the flow of Sales according to months with product ID descriptions

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(16,8))
num_graph = 2
sns.pointplot(x='Month', y='Sales', hue='Product_ID', 
                      data=df_grouped)
# Plotting the flow of Sales according to months with Country descriptions
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(16,8))
num_graph = 2
sns.pointplot(x='Month', y='Sales', hue='Country', 
                      data=df_grouped)
corr = df_grouped.corr()
corr
# Renaming the mismatching column indexes

df_expense.rename(columns={'Product_Type': 'Product_ID', 'Expense_Price': 'Sales'}, inplace=True)
# Joining the training data with expenses and do the differences
df_grouped1 = df_grouped.set_index(['Year','Month','Product_ID','Country'])
df_expense1= df_expense.set_index(['Year','Month','Product_ID','Country'])
df_diff = df_grouped1.join(df_expense1, how='outer', rsuffix='_').fillna(0)
df_grouped1['Sales'] = df_diff['Sales']- df_diff['Sales_']
# Index reset
df_grouped1 = df_grouped1.reset_index()

# top 5 rows
df_grouped1.head()
# Changing the sales data type
df_grouped1['Sales'] = df_grouped1['Sales'].astype(int)
df_grouped1.head()
df_grouped1.describe()
# Taking care of Categorical data in grouped1 country column
pd.get_dummies(df_grouped1, prefix=['Country'])
# Taking care of Categorical data in test country column
pd.get_dummies(test, prefix=['Country'])
# Defining a train function
def train_validate_split(df_grouped1, train_percent=.8, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df_grouped1.index)
    m = len(df_grouped1.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    training = df_grouped1.ix[perm[:train_end]]
    validate = df_grouped1.ix[perm[train_end:validate_end]]
    return training, validate
training, validate = train_validate_split(df_grouped1)

training
# Taking care of Categorical data in test country column
pd.get_dummies(training, prefix=['Country'])
# Taking care of Categorical data in test country column
pd.get_dummies(validate, prefix=['Country'])
# Splitting of Data Columns 
X_test = test.iloc[:, 1:4].values
Y_test = test.iloc[:, 5].values
Y_test
X_test
# Splitting of Data columns
X_train = df_grouped1.iloc[:, 0:3].values
Y_train = df_grouped1.iloc[:, 4].values
Y_train
X_train
# Splitting of Data columns
X_val = validate.iloc[:, 0:3].values
Y_val = validate.iloc[:, 4].values
Y_val
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 500, random_state = 0)
regressor.fit(X_train, Y_train)

# Predicting the values
y_pred = regressor.predict(X_test)
y_pred
np.savetxt("submission.csv", y_pred, delimiter=",")
my_imputer = Imputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.transform(X_test)
my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X_train, Y_train, verbose=False)
y_pred = my_model.predict(X_test)
y_pred
np.savetxt("submission1.csv", y_pred, delimiter=",")