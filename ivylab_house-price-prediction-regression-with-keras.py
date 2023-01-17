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
from tensorflow import keras

from tensorflow.keras import layers

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

from matplotlib import rcParams

%matplotlib inline

import seaborn as sns

import re
#load the train data

data_raw = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')



#load the test data

data_val = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#copy the data

data1 = data_raw.copy(deep = True)



#concat them for the data clean

data_clean = [data1, data_val]



#set option

pd.set_option('display.max_columns', data1.shape[1])



#make a overview of the train data

print(data_raw.info())

data_raw.sample(10)
print(data1.shape, data_val.shape)

data_raw.describe(include = 'all')
#show the data type condition

for dataset in data_clean:

    print(dataset.dtypes.value_counts())



data_clean[0].shape, data_clean[1].shape
#view the missing data condition



na_values = pd.concat([(data1.isnull().sum() /  data1.isnull().count())*100,

                        (data_val.isnull().sum() / data_val.isnull().count())*100], axis=1, keys=['Train', 'Test'], sort=False)

na_values[na_values.sum(axis = 1) > 0].sort_values(by = ['Train'], ascending = False)
#delete the feature that have too much missing data (>=50%)

missing_columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']



for dataset in data_clean:

    dataset.drop(missing_columns, axis = 1, inplace = True)

    

print('Train columns with null values:\n', data1.isnull().sum())

print("-"*10)



print('Test/Validation columns with null values:\n', data_val.isnull().sum())

print("-"*10)
data1.isnull().values.sum(), data_val.isnull().values.sum()
numeric_features = data1.dtypes[data1.dtypes != 'object'].index
medians = data1[numeric_features].median()

medians
#fill the na value in numerics columns:



data1[numeric_features] = data1[numeric_features].fillna(medians)

print('Train columns with null values:\n', data1.isnull().sum())

print("-"*10)
data1[numeric_features].isnull().values.sum()
#the same to the test data

numeric_features_val = data_val.dtypes[data_val.dtypes != 'object'].index

medians_val = data_val[numeric_features_val].median()



data_val[numeric_features_val] = data_val[numeric_features_val].fillna(medians_val)

print('Test columns with null values:\n', data_val.isnull().sum())

print("-"*10)
data_val[numeric_features_val].isnull().values.sum()
#preprocessing

target = data1['SalePrice']

train_Id = data1['Id']

data1 = data1.drop(['SalePrice'], axis = 1)

data1 = data1.drop(['Id'], axis = 1)



test_Id = data_val['Id']

data_val = data_val.drop(['Id'], axis = 1)



data1.shape, data_val.shape, target.shape
#get dummies



object_feature_1 = data1.dtypes[data1.dtypes == 'object'].index

object_feature_2 = data_val.dtypes[data_val.dtypes == 'object'].index



data1 = pd.get_dummies(data1, columns = object_feature_1, dummy_na = True)

data_val = pd.get_dummies(data_val, columns = object_feature_2, dummy_na = True)
columns1 = data1.columns

columns2 = data_val.columns



columns1.shape, columns2.shape
#take a look of the columns that test data doesnt have



s1 = set(columns1)

s2 = set(columns2)



s3 = s1-s2

s3
#add these columns to make train data and test data have the same shape



for feature in s3:

    data_val[feature] = 0

    

data1.columns.shape, data_val.columns.shape
#sorted the 2 dataFrame along the dictionary 

col = list(columns1)

col.sort()

col
#sort the train data

data1.reindex(col, axis = 'columns')
data_val.reindex(col, axis = 'columns')
#prepare the train and test data



X_train = data1.values.astype(np.float64, copy = False)

y_train = target.values





X_test = data_val.values.astype(np.float64, copy = False)



y_train = np.log(y_train)



print(X_train)

print('-'*10)

print(y_train)



train_Id, test_Id
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)



X_train, X_test
X_train.shape
X_test.shape
y_train.shape
#modeling



from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, BatchNormalization

import keras

from keras.optimizers import SGD

import graphviz

from keras.optimizers import Adam



def create_model(input_shape):

    model = Sequential()

    model.add(Dense(128, input_dim=input_shape, activation='sigmoid'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(16, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(1))

    

    optimizer = Adam(lr=0.005, decay=0.)

    model.compile(optimizer=optimizer,

             loss='mse',

             metrics=['mse'])

    return model

model = create_model(308)

model.summary()
model.fit(X_train, y_train, epochs = 150, batch_size = 32)
#prediction



prediction = model.predict(X_test)

prediction = np.exp(prediction)
submission = pd.DataFrame()

submission['Id'] = test_Id

submission['SalePrice'] = prediction

print('Saving prediction to output...')
submission.to_csv('prediction_regression.csv', index = False)

print("Done.")

print(submission)