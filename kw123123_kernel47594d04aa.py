import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

import pandas as pd

import datetime as dt

import seaborn as sns

from scipy import stats

%matplotlib inline 
train = pd.read_csv('data/train.csv',parse_dates=['datetime'])

test = pd.read_csv('data/test.csv',parse_dates=['datetime'])
train.info()
train['year']=train['datetime'].dt.year

train['month']=train['datetime'].dt.month

train['day']=train['datetime'].dt.day

train['hour']=train['datetime'].dt.hour

train['dayofweek']=train['datetime'].dt.dayofweek

train['minute']=train['datetime'].dt.minute

train['second']=train['datetime'].dt.second

train.shape
test['year']=test['datetime'].dt.year

test['month']=test['datetime'].dt.month

test['hour']=test['datetime'].dt.hour

test['dayofweek']=test['datetime'].dt.dayofweek

test.shape
categorical_feature_name = ["season","holiday","workingday","weather","dayofweek","month","year","hour"]
for var in categorical_feature_name:

    train[var]= train[var].astype("category")

    test[var]= test[var].astype("category")
feature_names = ["season","weather","temp","atemp","humidity","year","hour","dayofweek","holiday"

               ,"workingday"]

feature_names
X_train= train[feature_names]



print(X_train.shape)

X_train.head()
X_test= test[feature_names]



print(X_test.shape)

X_train.head()
label_name = "count"

y_train = train[label_name]



print(y_train.shape)

y_train.head()
from sklearn.metrics import make_scorer



def rmsle(predicted_values, actual_values , convertExp=True):

    

    if convertExp:

        predicted_values = np.exp(predicted_values),

        actual_values = np.exp(actual_values)

    

    predicted_values = np.array(predicted_values)#1

    actual_values = np.array(actual_values)



    log_predict = np.log(predicted_values + 1)#2

    log_actual = np.log(actual_values + 1)

    

    difference = log_predict - log_actual#3

    difference = np.square(difference)

    

    mean_difference = difference.mean()#4

    

    score = np.sqrt(mean_difference)#5

    

    return score

rmsle_scorer = make_scorer(rmsle)

rmsle_scorer
from sklearn.ensemble import GradientBoostingRegressor

gbn = GradientBoostingRegressor(n_estimators=4000, alpha=0.01);



y_train_log = np.log1p(y_train)

gbn.fit(X_train, y_train_log)



preds = gbn.predict(X_train)

score = rmsle(np.exp(y_train_log),np.exp(preds),False)

print ("RMSLE Value For Gradient Boost: ", score)
predsTest = gbn.predict(X_test)

fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(12,5)

sns.distplot(y_train,ax=ax1,bins=50)

sns.distplot(np.exp(predsTest),ax=ax2,bins=50)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
%time score = cross_val_score(gbn, X_train,y_train, cv=k_fold, scoring=rmsle_scorer)

score= score.mean()

print("Score={0:.5f}".format(score))
submission = pd.read_csv('data/sampleSubmission.csv')



submission



submission["count"]= np.exp(predsTest)



print(submission.shape)

submission.head()
submission.to_csv("data/Score_{0:.5f}_submission.csv".format(score), index=False)