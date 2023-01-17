import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import cross_val_score

import xgboost as xgb

import lightgbm as lgb

from catboost import CatBoostClassifier, Pool, cv

import warnings

warnings.filterwarnings(action="ignore")

import shap



import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm, skew





import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from sklearn import svm



print(tf.__version__)
train=pd.read_csv("../input/esunbank-creditcard-data/train.csv")

test=pd.read_csv("../input/esunbank-creditcard-data/test.csv")

credit=pd.concat([train, test], sort=False)

len_train=train.shape[0]



train_ID = train['txkey']

test_ID = test['txkey']



print(credit.dtypes.sort_values())

print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(test.shape))

credit.head()
correlation = train.corr()

correlation['fraud_ind'].sort_values(ascending=False).head(15)
credit.isnull().sum()[credit.isnull().sum()>0]
train.flbmk=train.flbmk.fillna("N")

test.flbmk=test.flbmk.fillna("N")



train.flg_3dsmk=train.flg_3dsmk.fillna("N")

test.flg_3dsmk=test.flg_3dsmk.fillna("N")



train.flg_3dsmk=train.fraud_ind.astype('int')

test.flg_3dsmk=train.fraud_ind.astype('int')



train.insfg=train.fraud_ind.astype('int')

test.insfg=train.fraud_ind.astype('int')



train.ecfg=train.fraud_ind.astype('int')

test.ecfg=train.fraud_ind.astype('int')



train.ovrlt=train.fraud_ind.astype('int')

test.ovrlt=train.fraud_ind.astype('int')



train.flbmk=train.fraud_ind.astype('int')

test.flbmk=train.fraud_ind.astype('int')



train.fraud_ind=train.fraud_ind.astype('int')



print("After fillna")

print("Training dataset null column",train.isnull().sum()[train.isnull().sum()>0])

print("testing dataset null column",test.isnull().sum()[test.isnull().sum()>0])
train_stats = train.describe()

train_stats = train_stats.transpose()



# Drop columns due to the whole values are same 

Drop_columns = train_stats['min'][train_stats['max'] == train_stats['min']].index

print("The Column which has all same value :\n",Drop_columns)
xtrain=train.drop("fraud_ind",axis=1)

ytrain=train['fraud_ind']

xtest=test

categorical_indices = np.where(xtrain.dtypes != np.float)[0]

# credit.iloc[:,categorical_indices].describe()

# credit=pd.concat([train, test], sort=False)
modelXGB= xgb.XGBClassifier(n_estimators=5000,nthread=4,max_depth=200,random_state=1)

scoresXGB=cross_val_score(modelXGB, xtrain, ytrain, scoring='accuracy', cv=5)

print("score : ",np.mean(scoresXGB))

modelXGB.fit(xtrain, ytrain)
xgb_predict = modelXGB.predict(xtest)

# make predictions which we will submit. 

output = pd.DataFrame({'txkey': test_ID,'fraud_ind': xgb_predict})

output.to_csv('submit_test.csv', index=False)