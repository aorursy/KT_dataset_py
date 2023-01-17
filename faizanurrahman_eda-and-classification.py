# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')
df.head()
df.isnull().sum()
df.isFraud.value_counts()
g = sns.countplot(x='isFraud', data=df)
g.set_yscale('log')
df.type.value_counts()
sns.countplot(x='type',data=df)
g = sns.countplot(x='type',hue='isFraud',data=df)
g.set_yscale('log')
df.amount.describe()
g = sns.boxplot(x='isFraud',y='amount',data=df)
g.set_yscale('log')
g = sns.boxplot(x='isFlaggedFraud',y='amount',data=df)
g.set_yscale('log')
g = sns.boxplot(x='isFraud',y='oldbalanceOrg',data=df)
g.set_yscale('log')
g = sns.boxplot(x='isFlaggedFraud',y='oldbalanceOrg',data=df)
g.set_yscale('log')
g = sns.boxplot(x='isFraud',y='oldbalanceDest',data=df)
g.set_yscale('log')
g = sns.boxplot(x='isFlaggedFraud',y='oldbalanceDest',data=df)
g.set_yscale('log')
dfc = df.copy(deep=True)
type_temp = pd.get_dummies(dfc['type'],drop_first=True)
dfc.drop(['type','nameOrig','nameDest'], axis = 1,inplace = True)
dfc = pd.concat([dfc,type_temp],axis=1)
dfc.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dfc.drop('isFraud',axis=1), 
                                                    dfc['isFraud'], test_size=0.30)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
