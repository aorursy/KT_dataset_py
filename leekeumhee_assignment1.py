import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("pandas version: ", pd.__version__)
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)
x_train=pd.read_csv('../input/inputdata/X_train.csv', sep=r'\s*,\s*',
                           header=0, encoding='utf-8', engine='python')
x_train=x_train.drop(['id'], axis=1)
x_train
x_train.shape
y_train=pd.read_csv('../input/inputdata/y_train.csv', sep=r'\s*,\s*',
                           header=0, encoding='utf-8', engine='python')
y_train=y_train.drop(['id'], axis=1)
y_train
x_test=pd.read_csv('../input/inputdata/X_test.csv', sep=r'\s*,\s*',
                           header=0, encoding='utf-8', engine='python')
x_test
x_id=x_test['id']
y_test=pd.read_csv('../input/inputdata/sample_submission.csv', sep=r'\s*,\s*',
                           header=0, encoding='utf-8', engine='python')
y_test
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc=StandardScaler()
sc.fit_transform(x_train)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit_transform(x_train).toarray()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=42, stratify=y_train, test_size=0.3)
gnb=GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
y_pred
y_pred.shape
y_test.shape
accuracy_score(y_test, y_pred)
submission=pd.DataFrame({'id': x_id, 'target': y_pred})
submission.to_csv("submission36.csv", index=False)
submission.shape
