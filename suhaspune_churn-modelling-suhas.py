# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
pd.set_option('display.max_columns', None)
dataset = pd.read_csv("../input/predicting-churn-for-bank-customers/Churn_Modelling.csv")
print(dataset.shape)
dataset.head()
dataset= dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis =1)
dataset.Exited.value_counts()
cat_var = [col for col in dataset.columns if dataset[col].dtype =='O']
cat_var
for col in ['Geography', 'Gender']:
    print(col,':')
    print( dataset[col].value_counts())
    print()
sns.pairplot(dataset, hue='Exited')
plt.figure(figsize =(22,12))
sns.heatmap(dataset.corr(), annot= True, cmap ='RdYlGn')
dataset.isnull().sum()


for col in ['CreditScore','Age','Balance','EstimatedSalary' ]:
    plt.figure(figsize=(14,8))
    plt.subplot(1,3,1)
    dataset[col].plot.hist(bins =30)
    plt.title(col)
    
    plt.subplot(1,3,2)
    stats.probplot(dataset[col], plot=pylab)
    plt.title(col)
    
    plt.subplot(1,3,3)
    dataset[col].plot.box()
    plt.title(col)
    plt.show()
                   
df =pd.concat([pd.get_dummies(dataset['Gender'], drop_first =True, prefix='Gender'),
              pd.get_dummies(dataset['Geography'], drop_first=True, prefix ='Geography'),
              dataset.drop(['Gender', 'Geography'], axis =1)], axis =1)
df.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('Exited', axis =1), df.Exited, test_size =0.2, random_state =0)
X_train.shape, X_test.shape
X_train.describe()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled= sc.fit_transform(X_train)
X_test_scalled = sc.transform(X_test)
from sklearn.metrics import roc_auc_score

import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict_proba(X_train)
print(roc_auc_score(y_train, y_pred[:,1]))

y_pred = xgb_model.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:,1]))

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
y_pred =ada.predict_proba(X_train)
print(roc_auc_score(y_train, y_pred[:,1]))

y_pred = ada.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:,1]))

from sklearn.ensemble import GradientBoostingClassifier
GBC =GradientBoostingClassifier()

GBC.fit(X_train, y_train)
y_pred =GBC.predict_proba(X_train)
print(roc_auc_score(y_train, y_pred[:,1]))

y_pred = GBC.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:,1]))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict_proba(X_train_scaled)
print(roc_auc_score(y_train, y_pred[:,1]))

y_pred = lr.predict_proba(X_test_scalled)
print(roc_auc_score(y_test, y_pred[:,1]))
