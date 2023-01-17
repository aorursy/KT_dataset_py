# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading the Train file into a dataframe and breaking up time column to diffrent columns to make it ready to be fed into our model

# First entry in the dataframe has a very odd behaviour. I felt better to remove it before creating model

df = pd.read_csv('../input/into-the-future/train.csv')

df = df.drop('id',axis=1)

df['date_conv'] = pd.to_datetime(df['time'], format='%Y-%m-%d')

df['hr'] = df['date_conv'].dt.hour

df['min'] = df['date_conv'].dt.minute

df['sec'] = df['date_conv'].dt.second

df
# Preparing X_train and y_train

X_train = df[['hr', 'min', 'sec', 'feature_1']]

y_train = df['feature_2']
# I am using sklearn's random forest estimator 

from sklearn.ensemble import RandomForestRegressor 

rfreg = RandomForestRegressor(n_estimators = 50)

rfreg.fit(X_train, y_train)
# loading test data into another dataframe 

df_2 = pd.read_csv('../input/into-the-future/test.csv')

df_2['date_conv'] = pd.to_datetime(df_2['time'], format='%Y-%m-%d')

df_2['hr'] = df_2['date_conv'].dt.hour

df_2['min'] = df_2['date_conv'].dt.minute

df_2['sec'] = df_2['date_conv'].dt.second

df_2
# Predicting values from test data and loading it into dataframe and then removing unneccesary columns

X_test = df_2[['hr', 'min', 'sec', 'feature_1']]

ypred = rfreg.predict(X_test)

df_2['feature_2'] = ypred

del df_2['sec']

del df_2['min']

del df_2['hr']

del df_2['date_conv']

df_2.to_csv('TERABXT_MOHIT')
# Visualizing train and test data sumultaneously

import matplotlib.pyplot as plt

%matplotlib notebook

plt.figure()

plt.plot(df['time'], df['feature_2'])

plt.plot(df_2['time'], ypred, 'r')