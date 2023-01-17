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
from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import GridSearchCV
df=pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.tail()
df.shape
df.info()
df.drop('customerID',axis=1,inplace=True)
df['TotalCharges'].replace(" ","0",regex=True,inplace=True)

df['TotalCharges']=pd.to_numeric(df['TotalCharges'])
X=df.iloc[:,:-1]

y=df.iloc[:,-1:]
X.shape
y.shape
X_encoded=pd.get_dummies(X,columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 

       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',

       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',

       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'

       ])
X_encoded.shape
y['Churn'].value_counts()
1869/7043*100
X_train,X_test,y_train,y_test=train_test_split(X_encoded,y,random_state=69,stratify=y)
xgb_clt=xgb.XGBClassifier(objective='binary:logistic',missing=None,seed=69)

xgb_clt.fit(X_train,y_train,

           early_stopping_rounds=10,

           eval_metric='aucpr',

           eval_set=[(X_test,y_test)])
plot_confusion_matrix(xgb_clt,

                     X_test,

                     y_test)
param_grid={'max_depth':[3,4,5],

           'learning_rate':[0.1,0.01,0.05],

           'gamma':[0,0.25,1.0],

           'reg_lamda':[0,1.0,10],

           'scale_pos_weight':[3,5]}

param_search=GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic',

                                                     seed=69,

                                                     subsample=0.9,

                                                     colsample_bytree=0.5),

                         param_grid=param_grid,

                         scoring='roc_auc',

                         n_jobs=10,

                         cv=3)

param_search.fit(X_train,y_train)
param_search.best_params_
param_grid={'max_depth':[1,2,3],

           'learning_rate':[0.05,0.01,0.005],

           'gamma':[1.0,1.25,1.50],

           'reg_lamda':[0],

           'scale_pos_weight':[3,3.5,4]}

param_search=GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic',

                                                     seed=69,

                                                     subsample=0.9,

                                                     colsample_bytree=0.5),

                         param_grid=param_grid,

                         scoring='roc_auc',

                         n_jobs=10,

                         cv=3)

param_search.fit(X_train,y_train)
param_search.best_params_
param_search.best_score_
clf_xgb=xgb.XGBClassifier(seed=69,

                         objective='binary:logistic',gamma=1.5,

                        learning_rate=0.05,

                        max_depth=3,

                        reg_lamda=0,

                        scale_pos_weight=3.5

                        )

clf_xgb.fit(X_train,y_train,

           early_stopping_rounds=10,

           eval_metric='aucpr',

            eval_set=[(X_test,y_test)]

           )
plot_confusion_matrix(clf_xgb,

                     X_test,y_test,

                     )
65/402*100