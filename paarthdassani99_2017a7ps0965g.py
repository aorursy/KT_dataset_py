# Import Dependencies

%matplotlib inline



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Machine learning

import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn import *



import warnings

warnings.filterwarnings('ignore')
# Import train & test data 

train = pd.read_csv('kaggle_data/train.csv')

test = pd.read_csv('kaggle_data/test.csv')
# removing features with 0 variance

#Select Model

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold() #Defaults to 0.0, e.g. only remove features with the same value in all samples



#Fit the Model

selector.fit(train)

X_varianceThreshold = train.loc[:, selector.get_support()]
X_varianceThreshold = X_varianceThreshold.drop(['id'], axis = 1)

X_varianceThreshold.describe()
X_varianceThreshold = X_varianceThreshold.sample(frac=1).reset_index(drop=True)

selected_df = X_varianceThreshold

X_train = selected_df.drop('label', axis=1) # data

y_train = selected_df.label # labels
best_feat = ['b90',

 'b80',

 'b31',

 'b3',

 'a5',

 'b22',

 'b50',

 'b7',

 'b84',

 'b55',

 'b27',

 'b20',

 'b78',

 'b0',

 'b62',

 'a0',

 'b29',

 'b69',

 'b38',

 'b23',

 'b77',

 'b34',

 'b87',

 'b76',

 'b93',

 'b48',

 'b86',

 'b19',

 'time',

 'b65',

 'b53',

 'b9',

 'b8',

 'b11',

 'b82',

 'b41',

 'b79',

 'b24',

 'b66',

 'b15']
for column in ['time', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'label']:

    if column in best_feat:

        print('true')

    else:

        best_feat.append(column)
len(best_feat)
#getting train to appropriate features

all_features = X_train.columns.tolist()

to_drop = []

for column in all_features:

    if column not in best_feat:

        to_drop.append(column)

X_train = X_train.drop(X_train[to_drop], axis=1)
X_train.describe()
final_id = test.id
best_feat.remove('label')
#getting test to appropriate features

all_features = test.columns.tolist()

to_drop = []

for column in all_features:

    if column not in best_feat:

        to_drop.append(column)

test = test.drop(test[to_drop], axis=1)
test.describe()
X_test = test
X_test.shape
#scale

from sklearn.preprocessing import RobustScaler

robust = RobustScaler(quantile_range = (0.2,0.8))

X_train = robust.fit_transform(X_train)

X_test = robust.transform(X_test)
# get results

reg = ensemble.RandomForestRegressor(random_state=0, n_estimators=500, n_jobs=-1)

reg_t = reg.fit(X_train,y_train)

y_pred_t = reg_t.predict(X_test)
y_pred_t
y_test = pd.Series(y_pred_t)

y_test
final_id
frame = { 'id': final_id, 'label': y_test } 

  

result = pd.DataFrame(frame)
result
result.to_csv("submission4.csv", index=False)