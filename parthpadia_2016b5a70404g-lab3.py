import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

num = LabelEncoder()

sns.set() # for plot styling
X_train = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

X_test = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
answer = X_test['custId']
X_train.info()
X_train['TotalCharges'] = pd.to_numeric(X_train['TotalCharges'],errors='coerce')

X_train['gender'] = num.fit_transform(X_train['gender'])

X_train['HighSpeed'] = X_train['HighSpeed'].replace({'No internet':0,'No':1,'Yes':2})

X_train['Married'] = num.fit_transform(X_train['Married'])

X_train['Children'] = num.fit_transform(X_train['Children'])

X_train['AddedServices'] = num.fit_transform(X_train['AddedServices'])

X_train['Channel1'] = X_train['Channel1'].replace({'No tv connection':0,'No':1,'Yes':2})

X_train['Channel2'] = X_train['Channel2'].replace({'No tv connection':0,'No':1,'Yes':2})

X_train['Channel3'] = X_train['Channel3'].replace({'No tv connection':0,'No':1,'Yes':2})

X_train['Channel4'] = X_train['Channel4'].replace({'No tv connection':0,'No':1,'Yes':2})

X_train['Channel5'] = X_train['Channel5'].replace({'No tv connection':0,'No':1,'Yes':2})

X_train['Channel6'] = X_train['Channel6'].replace({'No tv connection':0,'No':1,'Yes':2})

X_train['Subscription'].replace({'Monthly':1,'Annually':12,'Biannually':6},inplace=True)

X_train = pd.get_dummies(data=X_train,columns=['TVConnection','PaymentMethod'])

X_train.dropna(axis = 0,inplace=True)
X_test['TotalCharges'] = pd.to_numeric(X_test['TotalCharges'],errors='coerce')

X_test['gender'] = num.fit_transform(X_test['gender'])

X_test['HighSpeed'] = X_test['HighSpeed'].replace({'No internet':0,'No':1,'Yes':2})

X_test['Married'] = num.fit_transform(X_test['Married'])

X_test['Children'] = num.fit_transform(X_test['Children'])

X_test['AddedServices'] = num.fit_transform(X_test['AddedServices'])

X_test['Channel1'] = X_test['Channel1'].replace({'No tv connection':0,'No':1,'Yes':2})

X_test['Channel2'] = X_test['Channel2'].replace({'No tv connection':0,'No':1,'Yes':2})

X_test['Channel3'] = X_test['Channel3'].replace({'No tv connection':0,'No':1,'Yes':2})

X_test['Channel4'] = X_test['Channel4'].replace({'No tv connection':0,'No':1,'Yes':2})

X_test['Channel5'] = X_test['Channel5'].replace({'No tv connection':0,'No':1,'Yes':2})

X_test['Channel6'] = X_test['Channel6'].replace({'No tv connection':0,'No':1,'Yes':2})

X_test['Subscription'].replace({'Monthly':1,'Annually':12,'Biannually':6},inplace=True)

X_test = pd.get_dummies(data=X_test,columns=['TVConnection','PaymentMethod'])

X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean(),inplace=True)
X = X_train.drop(columns=['Satisfied','Internet','custId'])

y = X_train['Satisfied']

X_test = X_test.drop(columns=['Internet','custId'])
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scal_cols = ['TotalCharges','tenure','MonthlyCharges']



X.loc[:,  scal_cols] = scaler.fit_transform(X[scal_cols])
scal_cols = ['TotalCharges','tenure','MonthlyCharges']

scaler = MinMaxScaler()



X_test.loc[:,  scal_cols] = scaler.fit_transform(X_test[scal_cols])
from sklearn.cluster import KMeans

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

kmeans = KMeans(n_clusters=2).fit(X)

y_pred = kmeans.predict(X)

acc = roc_auc_score(y,y_pred)

y_test = kmeans.predict(X_test)
answer = pd.DataFrame(answer)

answer['Satisfied']=y_test

answer.to_csv('Kmeans_drop_1.csv',encoding='utf-8',index=False)