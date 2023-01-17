import os

print((os.listdir('../input/')))
import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score,accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV,cross_val_score

from sklearn.model_selection import StratifiedKFold 

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from matplotlib import pyplot

from xgboost import XGBClassifier

df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
test_index=df_test['Unnamed: 0'] #copying test index for late
df_train.head()
df_train.info()

print('_'*40)

df_test.info()
df_train[["V14", "Class"]].groupby(['V14'], as_index=False).mean().sort_values(by='Class', ascending=False)
def f(X):

    X['V17']=X.V3/X.V1

    X['V18']=(X.V14==-1)*1+(X.V16==3)*1

    X['V19']=(X.V7==1)*1+(X.V11==8)*1  

    X['V20']=(X.V1+X.V2+X.V3+X.V4+X.V8)/X.V10

    X['V21']=((X.V14==-1)*1+(X.V16==3)*1)/X.V10

    X['V22']=((X.V7==1)*1+(X.V11==8)*1)/X.V11

    X['V23']=0

    X.loc[X['V20'] == 1, 'V23'] = 1

    print(X.head())

    return X
df_train=f(df_train)
df_train[["V23", "Class"]].groupby(['V23'], as_index=False).mean().sort_values(by='Class', ascending=False)
train_X = df_train.loc[:, 'V1':'V16']

train_y = df_train.loc[:, 'Class']

train_X = f(train_X)
model = XGBClassifier(eval_metric='auc',learning_rate=0.069,n_estimators=261,gamma=0.5,max_depth=5,subsample=0.8,colsample_bytree=0.6,colsample_bynode=0.7,colsample_bylevel=0.9,min_child_weight=1.3,reg_lambda=0.7)

model.fit(train_X,train_y)
df_test = df_test.loc[:, 'V1':'V16']

df_test=f(df_test)

pred = model.predict_proba(df_test)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('output.csv', index=False)