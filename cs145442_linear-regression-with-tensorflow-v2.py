# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting library
import seaborn as sns # seaborn plotting library

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split # to split training data

import tensorflow as tf # tensorflow library
import tensorflow.compat.v2.feature_column as fc # tensorflow helper
# Importing the data
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
y_train = train_df.pop('SalePrice')
# Let's get the list of all the Numerical and Categorical Columns
NUMERIC_COLUMNS = train_df._get_numeric_data().columns.tolist()
CATEGORICAL_COLUMNS = list()
for cols in train_df.columns.tolist():
    if cols not in NUMERIC_COLUMNS:
        CATEGORICAL_COLUMNS.append(cols)
# Cleaning the NaN Values
for feature_name in train_df.columns:
    num_nan = train_df[feature_name].isnull().sum()
    if num_nan > 0:
        print(feature_name + (' : N ' if feature_name in NUMERIC_COLUMNS else ' : C ') + ' : ' + str(num_nan) )
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(0)
train_df['Alley'] = train_df['Alley'].fillna('NA')
train_df['MasVnrType'] = train_df['MasVnrType'].fillna('None')
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)
train_df['BsmtQual'] = train_df['BsmtQual'].fillna('NA')
train_df['BsmtCond'] = train_df['BsmtCond'].fillna('NA')
train_df['BsmtExposure'] = train_df['BsmtExposure'].fillna('NA')
train_df['BsmtFinType1'] = train_df['BsmtFinType1'].fillna('NA')
train_df['BsmtFinType2'] = train_df['BsmtFinType2'].fillna('NA')
train_df['Electrical'] = train_df['Electrical'].fillna('SBrkr')
train_df['FireplaceQu'] = train_df['FireplaceQu'].fillna('NA')
train_df['GarageType'] = train_df['GarageType'].fillna('NA')
train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(0)
train_df['GarageFinish'] = train_df['GarageFinish'].fillna('NA')
train_df['GarageQual'] = train_df['GarageQual'].fillna('NA')
train_df['GarageCond'] = train_df['GarageCond'].fillna('NA')
train_df['PoolQC'] = train_df['PoolQC'].fillna('NA')
train_df['Fence'] = train_df['Fence'].fillna('NA')
train_df['MiscFeature'] = train_df['MiscFeature'].fillna('NA')
# Splitting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(train_df, y_train, test_size=0.3, random_state=42)
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = train_df[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_function = make_input_fn(X_train, y_train)
test_input_function = make_input_fn(X_test, y_test)
ds = make_input_fn(X_train, y_train, batch_size=10)()
for feature_batch, label_batch in ds.take(1):
  print('Some feature keys:', list(feature_batch.keys()))
  print()
  print('A batch of class:', feature_batch['MiscFeature'].numpy())
  print()
  print('A batch of Labels:', label_batch.numpy())
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_function)
result = linear_est.evaluate(test_input_function)

clear_output()
print(result)