import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv')
#CONVERTING ONE HOT ENCODING INTO LABEL ENCODING



train['agent'] = (train['a1'] + 2*train['a2'] + 3*train['a3'] + 4*train['a4'] + 5*train['a5'] + 6*train['a6'])

train = train.drop(['a0','a1','a2','a3','a4','a5','a6'],axis=1)



#these columns dropped based on low importance features of previous model of random forest

drop_cols = ['b2', 'b5', 'b10', 'b12', 'b16', 'b17', 'b23', 'b26', 'b28', 'b35'

             , 'b36', 'b42', 'b44', 'b54', 'b58', 'b60', 'b61', 'b63', 'b64', 

             'b68', 'b71', 'b72', 'b75', 'b81', 'b83', 'b88', 'b89']



train = train.drop(drop_cols,axis=1)
train.corr()
#dropping columns with correlation <=0.02 with the label, to remove irrelevant features, but don't want to remove time



cor_target = train.corr().abs()['label']

remove = cor_target[cor_target<=0.02]

remove = remove.drop(labels = 'time',axis=0)

remove.index
train = train.drop(remove.index.tolist(),axis=1)
X = train.drop('label',axis=1)

y = train['label']
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
##Attempt to use adaboostregressor with 50,100,150,200,250 estimators, choose one with lowest rmse
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV
#below has n_estimators  = 50
#ada50 = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = 50,random_state = 42)

#ada50.fit(X_train, y_train)
#y_pred_ada = ada50.predict(X_val)



#print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred_ada))  

#print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred_ada))  

#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred_ada)))
#below has n_estimators  = 100
#ada100 = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = 100,random_state = 42)

#ada100.fit(X_train, y_train)
#y_pred_ada = ada100.predict(X_val)



#print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred_ada))  

#print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred_ada))  

#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred_ada)))
#below has n_estimators  = 150
#ada150 = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = 150,random_state = 42)

#ada150.fit(X_train, y_train)
#y_pred_ada = ada150.predict(X_val)



#print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred_ada))  

#print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred_ada))  

#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred_ada)))
#below has n_estimators  = 200
#da200 = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = 200,random_state = 42)

#ada200.fit(X_train, y_train)
#y_pred_ada = ada200.predict(X_val)



#print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred_ada))  

#print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred_ada))  

#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred_ada)))
#below has n_estimators  = 250
ada250 = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = 250,random_state = 42)

ada250.fit(X_train, y_train)
y_pred_ada = ada250.predict(X_val)



print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred_ada))  

print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred_ada))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred_ada)))
X_test = pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')

X_test['agent'] = (X_test['a1'] + 2*X_test['a2'] + 3*X_test['a3'] + 4*X_test['a4'] + 5*X_test['a5'] + 6*X_test['a6'])

X_test = X_test.drop(['a0','a1','a2','a3','a4','a5','a6'],axis=1)

X_test = X_test.drop(drop_cols,axis = 1)

X_test = X_test.drop(remove.index.tolist(),axis=1)



X_test
#use the estimators that gave best result for validation set of rmse.

#best result = 1.320



y_pred = ada250.predict(X_test)

y_pred = np.asarray(y_pred)



sub = pd.DataFrame(y_pred)

sub.index = sub.index+1

sub.to_csv('sub_5_ada_250_estimators.csv')

df = pd.read_csv('sub_5_ada_250_estimators.csv')



df.rename(columns={'Unnamed: 0':'id','0':'label'}, inplace=True)

df.to_csv('sub_5_ada_250_estimators.csv',index=False)