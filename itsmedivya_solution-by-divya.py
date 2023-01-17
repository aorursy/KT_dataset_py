# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/health-diagnostics-train.csv', na_values="#NULL!")
df.head() 
df.dtypes
df.columns[df.isnull().any()].tolist()
# Checking if columns can be read as expected datatype

new_series = pd.to_numeric(df['income'], errors="coerce")
new_series.isnull().unique()
df1 = df.dropna(axis=0)
sns.heatmap(df1.corr());
print(df1.corr())
sns.lineplot(y='fam-history', x='income', data=df1, ci=None);
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
# By looking at corr matrix and also 
feature_cols = ['suppl','mat-illness','env','fam-history', 'lifestyle']

X = df1[feature_cols]
y = df1.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test) # prediction via Log Reg
roc=roc_auc_score(y_test, logreg.predict_proba(X_test)[:,1])
roc
logreg.score(X_test, y_test)
yn = np.where((y_train == 0), 0, 1)
scaler = StandardScaler()
X_std = scaler.fit_transform(X_train)
# Create decision tree classifer object
clf = LogisticRegression(random_state=0, class_weight='balanced')

# Train model
model = clf.fit(X_std, yn)
y_pred = model.predict(X_test) # prediction via Log Reg
roc=roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
roc
df_test = pd.read_csv('../input/health-diagnostics-test.csv', na_values="#NULL!")
df_test.head() 
df_test = df_test.fillna(value=0)
pred = logreg.predict(df_test[feature_cols])
prediction = pd.DataFrame(pred, columns=['index,target']).to_csv('prediction06.csv')
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, random_state=20, stratify = y)
y_pred = logreg.predict(X_train1)
print(roc_auc_score(y_test1, logreg.predict_proba(X_test1)[:,1]))
print(roc_auc_score(y_train1, logreg.predict_proba(X_train1)[:,1]))
logreg.score(X_test1, y_test1)
logreg.score(X_train1, y_train1)
sns.lineplot(y='target', x='suppl', data=df1);
sns.lineplot(y='fam-history', x='mat-illness', data=df1);
sns.lineplot(y='target', x='mat-illness', data=df1);
sns.lineplot(y='target', x='env', data=df1);
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics

# rfreg = RandomForestClassifier(n_estimators=100, max_features=3, oob_score=True, random_state=1, class_weight = 'balanced')
# feature_cols = ['suppl','env','fam-history', 'mat-illness']
# X = df1[feature_cols]
# y = df1.target
# rfreg.fit(X, y)

# y_pred = rfreg.predict(X_train1)
# # np.sqrt(metrics.mean_squared_error(y_train1, y_pred))
# roc=roc_auc_score(y_train1, y_pred)
# roc
# pd.DataFrame({'feature':feature_cols, 'importance':rfreg.feature_importances_}).sort_values(by='importance')
# pred = rfreg.predict(df_test[feature_cols])
# y_pred_new = []
# for i in y_pred:
#     if i > 0.5:
#         y_pred_new.append(1)
#     else:
#         y_pred_new.append(0)
# prediction = pd.DataFrame(y_pred_new, columns=['index,target']).to_csv('prediction04.csv')