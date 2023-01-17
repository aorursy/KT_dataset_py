import sys

!{sys.executable} -m pip install --ignore-installed imblearn
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")

df_train.head()
df_submission = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/sample_submission.csv")

df_submission
df_train.info()
print(df_train.isnull().sum())
df_train.fillna(value=df_train.mean(),inplace=True)
print(df_train.isnull().sum())
df_train.isnull().any().any()
df_train.columns
numerical_features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',

       'feature6', 'feature7', 'feature8', 'feature9', 'feature10',

       'feature11']

categorical_features = ['type']

X = df_train[numerical_features+categorical_features]

y = df_train['rating']
type_val = {'old':0,'new':1}

X['type'] = X['type'].map(type_val)

X.head()
y.value_counts()
y.value_counts()
df_train_new = df_train
df_train['rating'].value_counts()
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=42,stratify=y)
from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

X_val[numerical_features] = scaler.transform(X_val[numerical_features])  



X_train[numerical_features].head()
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from imblearn.ensemble import BalancedRandomForestClassifier

#reg =  RandomForestClassifier(class_weight='balanced',n_estimators=400,max_depth=15).fit(X_train, y_train)

#clf = BalancedRandomForestClassifier(random_state=0,max_depth=5,n_estimators=500).fit(X_train, y_train)
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

reg_ada = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=25,max_features=5),random_state=42).fit(X_train,y_train)
from sklearn.ensemble import RandomForestRegressor

#RF_reg = RandomForestRegressor(n_estimators=300,max_depth=30,random_state=42).fit(X_train, y_train)

y_pred = reg_ada.predict(X_val)
'''import math as m

def floattoint(x):

    for i in range(len(x)):

        if(x[i]<3):

            x[i] = m.ceil(x[i])

        else:

            x[i] = m.floor(x[i])

    return x'''
print(y_pred)
#y_pred = floattoint(y_pred)

#y_pred = y_pred.astype('int64')

#y_pred
y_pred = np.array(y_pred)

y_pred = y_pred.round()

y_pred = [int(i) for i in y_pred]
from sklearn.metrics import mean_squared_error



from math import sqrt



rmse = sqrt(mean_squared_error(y_pred, y_val))



print(rmse)
reg_full = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=25,max_features=5),random_state=42).fit(X, y)
df_test = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")

df_test.head()
df_test.info()
df_test.fillna(value=df_test.mean(),inplace=True)
df_test.isnull().any().any()
numerical_features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',

       'feature6', 'feature7', 'feature8', 'feature9', 'feature10',

       'feature11']

categorical_features = ['type']

X_test = df_test[numerical_features+categorical_features]

type_val = {'old':0,'new':1}

X_test['type'] = X_test['type'].map(type_val)

X_test.head()
y_test = reg_full.predict(X_test)
y_test = np.array(y_test)

y_test = y_test.round()

y_test = [int(i) for i in y_test]
df_submission.columns
df_new = pd.DataFrame(columns = ['id', 'rating'])
df_new.head()
df_new["id"] = df_test["id"]

df_new["rating"] = y_test
df_new.to_csv("sub8.csv",index = False)