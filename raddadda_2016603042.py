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
train['year']=train['datetime'].dt.year

train['month']=train['datetime'].dt.month

train['day']=train['datetime'].dt.day

train['hour']=train['datetime'].dt.hour

train['dayofweek']=train['datetime'].dt.dayofweek

train.shape
test['year']=test['datetime'].dt.year

test['month']=test['datetime'].dt.month

test['hour']=test['datetime'].dt.hour

test['dayofweek']=test['datetime'].dt.dayofweek

test.shape
categorical_features = ["dayofweek","month","year","hour",

                    "season","holiday","workingday","weather"]
for part in categorical_features:

    train[part]= train[part].astype("category")

    test[part]= test[part].astype("category")
features = ["temp","atemp","humidity","workingday","hour","year",

            "dayofweek","weather","season","holiday"]

features
X_train= train[features]



print(X_train.shape)

X_train.head()
X_test= test[features]



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

    

    predicted_values = np.array(predicted_values)

    actual_values = np.array(actual_values)

   





    log_predict = np.log(predicted_values + 1)

    log_actual = np.log(actual_values + 1)

    

    difference = log_predict - log_actual

    difference = np.square(difference)

    

    mean_difference = difference.mean()

   

    score = np.sqrt(mean_difference)

    

    return score



RMSLE_Score = make_scorer(rmsle)

RMSLE_Score
from sklearn.ensemble import RandomForestRegressor

Ran_For_Model = RandomForestRegressor(n_estimators=100)



y_train_log= np.log1p(y_train)

Ran_For_Model.fit(X_train,y_train_log)

predicts= Ran_For_Model.predict(X_train)



print("RMSLE Value For Random Forest:", rmsle(np.exp(y_train_log),np.exp(predicts),False))
predicts_test = Ran_For_Model.predict(X_test)

fig,(ax1,ax2)=plt.subplots(ncols=2)

fig.set_size_inches(12,5)

sns.distplot(y_train,ax=ax1,bins=50,color='r')

sns.distplot(np.exp(predicts_test),ax=ax2,bins=50,color='g')
submission = pd.read_csv('data/sampleSubmission.csv')



submission["count"]= np.exp(predicts_test)



submission.head()
submission.to_csv("data/2016603042 submission.csv".format(score), index=False)