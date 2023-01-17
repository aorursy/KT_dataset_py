import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv("/kaggle/input/costa-rican-household-poverty-prediction/train.csv")
df_test = pd.read_csv("/kaggle/input/costa-rican-household-poverty-prediction/test.csv")

df_train.shape, df_test.shape
df_train.head()
df_test.head()
df_train.info()
df_test.info()
df_test.columns[df_test.dtypes==object]
df_train.columns[df_train.dtypes==object]
df_train.describe()
df_test.describe()
df_train.select_dtypes('object').head()
df_train.isna().sum()
df_test.isna().sum()
df_train.columns[df_train.isna().sum()!=False]
df_test.columns[df_test.isna().sum()!=False]
print(df_train['v2a1'].isna().sum())
print(df_train['v18q1'].isna().sum())
print(df_train['rez_esc'].isna().sum())
print(df_train['meaneduc'].isna().sum())
print(df_train['SQBmeaned'].isna().sum())
df_train['v2a1'] = df_train['v2a1'].fillna(0)
df_test['v2a1'] = df_test['v2a1'].fillna(0)
df_train['v18q1'] = df_train['v18q1'].fillna(0)
df_test['v18q1'] = df_test['v18q1'].fillna(0)
df_train[['meaneduc', 'SQBmeaned']].describe()
df_test[['meaneduc', 'SQBmeaned']].describe()
df_train['meaneduc'].fillna(0)
df_train['SQBmeaned'].fillna(0)
df_test['meaneduc'].fillna(0)
df_test['SQBmeaned'].fillna(0)
df_train['rez_esc'].fillna(0)
df_test['rez_esc'].fillna(0)
import matplotlib.pyplot as plt
df_train['Target'].plot.hist()
df_train['Target'].value_counts()
df_train.head()
df_train['edjefe'].value_counts()
df_train['edjefa'].value_counts()
df_train['dependency'].value_counts()
mapping = {"yes": 1, "no": 0}

for df in [df_train, df_test]:
    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)
    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)

df_train[['dependency', 'edjefa', 'edjefe']].describe()
df_test.info()
df_train.drop(['Id'],axis=1)
df_train.drop(['idhogar'],axis=1)
df_train.select_dtypes('object').head()

df_train = df_train.drop(columns = ['SQBmeaned', 'agesq'])
df_train.shape
df_test = df_test.drop(columns = ['SQBmeaned', 'agesq'])
df_test.shape
df_train.info()
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
X=df_train.drop(['Id', 'idhogar', 'Target'], axis=1)

y=df_train['Target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

X_train.shape, y_train.shape
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)
print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
import lightgbm as lgb

lgb = lgb.LGBMClassifier()
lgb.fit(X_train, y_train)
preds = lgb.predict(X_test)
print(accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))
z = pd.Series(lgb.predict(X_test),name="Target")
df_entrega = pd.concat([df_test.Id,z], axis=1)
df_entrega.to_csv("/kaggle/working/submission.csv",index=False)