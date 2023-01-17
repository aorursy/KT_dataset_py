import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

pd.set_option('display.max_columns', None)
X_train=pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

y_train=pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

X_test=pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
X_train.shape,y_train.shape,X_test.shape
X_train.head()
y_train.head()
X_test.head()
index=X_test['sig_id']

X_test.drop('sig_id',axis=1,inplace=True)

y_train.drop('sig_id',axis=1,inplace=True)

X_train.drop('sig_id',axis=1,inplace=True)

X_train.shape,y_train.shape,X_test.shape
X_train.shape,y_train.shape
prime = pd.Series(index,name="sig_id")

prime.head()
X_train.head()
X_train.shape,y_train.shape
from sklearn.model_selection import train_test_split

X_trainc,X_evalc,y_trainc,y_evalc=train_test_split(X_train,y_train, train_size = .99,random_state =333)
X_trainc.shape,X_evalc.shape,y_trainc.shape,y_evalc.shape


import sklearn.metrics

result=pd.DataFrame()

from catboost import CatBoostClassifier

model=CatBoostClassifier(task_type='GPU',boosting_type='Plain',bootstrap_type='Bayesian',iterations=500,use_best_model=True,

                         devices='0:1')

for i in range(len(y_trainc.columns)):

    model.fit(X_trainc,y_trainc.iloc[:,i],cat_features=[0,2],eval_set=(X_evalc, y_evalc.iloc[:,i]))

    pred = model.predict(X_test)

    result.insert(i,y_trainc.columns[i],list(pred))
result.head()
submission = pd.concat([prime,result],axis = 1)

submission.to_csv("submission.csv",index=False)

submission.head()