import numpy as np
import pandas as pd

import string
import re

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight') 

import gc
gc.enable()

import warnings
warnings.filterwarnings('ignore')
PATH = 'data/realestate/'
test = pd.read_table(PATH + 'test.csv'  ,sep=',')
train = pd.read_table(PATH + 'train.csv'  ,sep=',')
train.head(10)

train.info(verbose=False)
test.info(verbose=False)

train.isnull().sum()
test.isnull().sum()
import missingno as msno

def missingno_matrix(df, figsize=(10,8)):
    ''' Missing Value visualization by matrix plot

        Parameters:
        -----------
        df: DataFrame

        Return: matrix plot
        -----------
    '''

    missingValueColumns = df.columns[df.isnull().any()].tolist()
    msno.matrix(df[missingValueColumns],width_ratios=(10,1),\
            figsize=figsize,color=(0, 0, 0),fontsize=12, sparkline=True, labels=True)
    plt.show()
    
missingno_matrix(train)
sns.distplot(train['SalePrice'], bins=100, kde=False)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()
sns.distplot(train[train['SalePrice']<=500000]['SalePrice'], bins=100, kde=False)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()
train.SalePrice.nunique()
train.SalePrice.describe() 

train[train['SalePrice']==34900].head(20)
from sklearn import preprocessing

# Single out the features
train_features = train.loc[:, 'MSSubClass':'SaleCondition']
test_features = test.loc[:, 'MSSubClass':'SaleCondition']

# Group data to ensure same dimensions post one-hot encoding
all_data = pd.concat([train_features, test_features])

all_data = pd.get_dummies(all_data)

# Handle missing values
imputer = preprocessing.Imputer()

all_data = imputer.fit_transform(all_data)

# Separate data
train_features = all_data[:train.shape[0]]
test_features = all_data[train.shape[0]:]

train_target = train[['SalePrice']]

print(train_features.shape)
print(test_features.shape)

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from math import sqrt

model = ensemble.GradientBoostingRegressor(n_estimators=15000, max_depth=5, min_samples_leaf=15, min_samples_split=10, learning_rate=0.01, loss='huber', random_state=5)

# Reshape train_target to be a 1d array
train_target = train_target.as_matrix().flatten()

# Fit model
model.fit(train_features, train_target)
# Make predictions with model
target_predictions = model.predict(test_features)

target_predictions = np.reshape(target_predictions, -1)

# Prepare solution
solution = pd.DataFrame({"id":test.Id, "SalePrice":target_predictions})

solution.to_csv('submission1.csv', index=False)
