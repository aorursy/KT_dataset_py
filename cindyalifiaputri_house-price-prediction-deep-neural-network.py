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
# Linear algebra
import numpy as np

# Data preprocessing csv
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Splitting data train and validation
from sklearn.model_selection import train_test_split

# Tensorflow for modelling
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout
print("Tensorflow version : {}".format(tf.__version__))
# Load data training
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

# Load data testing
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# split data between data train and the label
train = df_train.drop(['SalePrice'], axis=1)
y = df_train['SalePrice']
# concat between data train and data test
data=pd.concat([train, df_test], axis=0)
# print each shape of data
print("The shape of data train : {} ".format(df_train.shape))
print("The shape of data test : {} ".format(df_test.shape))
print("The shape of all data (label excluded) : {} ".format(data.shape))
# Check numercial data
data.describe()
# Check numercial data
df_train.describe(include=['object', 'bool'])
df_train.sort_values(by='SalePrice', ascending=False).head()
# Check total price of each year 
df_analysis = {'Year Sold':['2006', '2007', '2008', '2009', '2010'],
               'Mean Sale Price': [round(df_train.query('YrSold == 2006').SalePrice.mean()),
                            round(df_train.query('YrSold == 2007').SalePrice.mean()),
                            round(df_train.query('YrSold == 2008').SalePrice.mean()),
                            round(df_train.query('YrSold == 2009').SalePrice.mean()),
                            round(df_train.query('YrSold == 2010').SalePrice.mean())],
               'Min Sale Price': [df_train.query('YrSold == 2006').SalePrice.min(),
                                  df_train.query('YrSold == 2007').SalePrice.min(),
                                  df_train.query('YrSold == 2008').SalePrice.min(),
                                  df_train.query('YrSold == 2009').SalePrice.min(),
                                  df_train.query('YrSold == 2010').SalePrice.min()],
               'Max Sale Price': [df_train.query('YrSold == 2006').SalePrice.max(),
                                  df_train.query('YrSold == 2007').SalePrice.max(),
                                  df_train.query('YrSold == 2008').SalePrice.max(),
                                  df_train.query('YrSold == 2009').SalePrice.max(),
                                  df_train.query('YrSold == 2010').SalePrice.max()]}
df_temp = pd.DataFrame(df_analysis, index=['1', '2', '3', '4', '5'])
df_temp
# average price of house sales
sns.set(style='whitegrid')
sns.lineplot(x="Year Sold", y="Mean Sale Price", 
                  data=df_temp)
# Let's see the correlation between each column (numerical column) 
# give the most correlation with saleprice
for i in range(0, len(df_train.columns), 4):
    sns.pairplot(data=df_train,
                x_vars=df_train.columns[i:i+4],
                y_vars=['SalePrice'])
# Find the percentage of NaN value in data
# check target nan value in the data 
combined = data.copy()
nan_percentage = combined.isnull().sum().sort_values(
    ascending=False) / combined.shape[0]
missing_val = nan_percentage[nan_percentage > 0]

plt.figure(figsize=(9,7))
sns.barplot(x=missing_val.index.values, 
            y=missing_val.values * 100, 
            palette="Reds_r");
plt.title("Percentage of missing values in data");
plt.ylabel("%");
plt.xticks(rotation=90);
# Delete columns that has percentage NaN value above 50%
data.drop(columns=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],
          axis=1, inplace=True)
# Delete columns 'Id' because that is not important to the training process
data.drop(columns=['Id'], inplace=True, axis=1)
# Selecting columns that contain NaN value
nan_col = data.columns[data.isnull().any()].tolist()
data[nan_col].head(5)
# Fill NaN value with dtype is not Object with the average of the data
df=data.fillna(data.mean())
# Find the percentage of NaN value in data
# check target nan value in the data 
combined = df.copy()
nan_percentage = combined.isnull().sum().sort_values(
    ascending=False) / combined.shape[0]
missing_val = nan_percentage[nan_percentage > 0]

plt.figure(figsize=(9,7))
sns.barplot(x=missing_val.index.values, 
            y=missing_val.values * 100, 
            palette="Reds_r");
plt.title("Percentage of missing values in data");
plt.ylabel("%");
plt.xticks(rotation=90);
# Selecting columns that still contain NaN value
nan_col = df.columns[df.isnull().any()].tolist()
# Replace NaN value on each categorical column with modus of the data
for column in nan_col:
    df[column].fillna(df[column].mode()[0], inplace=True)
# Select all numerical columns
numerical_columns=df.select_dtypes(exclude=['O']).columns
# Print 5 random numerical columns that we are going to normalize it into 0-1
df[numerical_columns].sample(5, random_state=99)
from sklearn.preprocessing import MinMaxScaler
# build function to data normalization
def normalize_data(df):
  # Select all numerical columns
  numerical_columns=df.select_dtypes(exclude=['O']).columns

  scaling=MinMaxScaler()
  df[numerical_columns]=scaling.fit_transform(df[numerical_columns])
  return df
# normalize data
df_norm = normalize_data(df)
# Check if the data already transformed
df_norm[numerical_columns].sample(5, random_state=99)
df_norm.sample(5, random_state=99)
# Select all categorical columns
categorical_columns=df_norm.select_dtypes(include=['O']).columns
print(categorical_columns)
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import sklearn as s
# Check sklearn version 
print(s.__version__)
# Build object to column_transformer
column_transformer = make_column_transformer(
    (OneHotEncoder(), categorical_columns),
    remainder='passthrough'
)
# Transform data 
df_encoded = column_transformer.fit_transform(df_norm)
df_encoded
# Change data type from sparse into DataFrame
import scipy.sparse
df_encode = pd.DataFrame.sparse.from_spmatrix(df_encoded)
df_encode.sample(5, random_state=99)
# Bring back data training and data testing 
X_train = df_encode.iloc[:-len(df_test), :]
X_test = df_encode.iloc[len(df_train):, :]
# Make sure that we split data with the right shape and place
print("\nOld data")
print("The shape of data train : {} ".format(df_train.shape))
print("The shape of data test : {} ".format(df_test.shape))

print("\nNew data")
print("The shape of data train : {} ".format(X_train.shape))
print("The shape of data test : {} ".format(X_test.shape))

print("The shape of class : {} ".format(y.shape))
# Make an object to a model
model = Sequential()

# The Input Layer 
model.add(Dense(512, kernel_initializer='normal',
                   input_dim = X_train.shape[1], 
                   activation='relu'))

# The Hidden Layers 
model.add(Dense(216, kernel_initializer='normal',activation='relu'))
model.add(Dense(216, kernel_initializer='normal',activation='relu'))
model.add(Dense(128, kernel_initializer='normal',activation='relu'))
model.add(Dense(64, kernel_initializer='normal',activation='relu'))
model.add(Dense(8, kernel_initializer='normal',activation='relu'))

# The Output Layer 
model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network 
model.compile(loss='mean_absolute_error', 
                 optimizer='adam', 
                 metrics=['mean_absolute_error'])
model.summary()
# Build callback object to stop learning
filepath = 'model.{epoch:02d}-{val_loss:.2f}.h5'
callback = ModelCheckpoint(filepath, 
                             monitor='val_loss', 
                             verbose = 1, 
                             save_best_only = True, 
                             mode ='auto')
# Train a model
model.fit(X_train, y, 
             epochs=1000, 
             batch_size=32, 
             validation_split = 0.25, 
             callbacks=[callback])
# Predict with the data testing 
predictions = model.predict(X_test)
# Load submission file
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission['SalePrice'] = predictions
submission.head(6)
# Save the prediction
submission.to_csv('Submission.csv', index=False)
