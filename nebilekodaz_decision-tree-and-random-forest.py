# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df=pd.read_csv('../input/loan_data.csv')
df.head()
df.info()
df.describe()
plt.figure(figsize=(10,6))
df[df['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
df[df['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')

plt.legend()
plt.xlabel('FICO')
plt.figure(figsize=(10,6))
df[df['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
df[df['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')

plt.legend()
plt.xlabel('FICO')
fig = plt.figure(figsize=(10,6))
sns.set_context("paper", font_scale=1)
sns.countplot(x='purpose',data=df, hue='not.fully.paid')
sns.jointplot(x='fico',y='int.rate',data=df,color='purple')
sns.lmplot(x='fico',y='int.rate',data=df,col='not.fully.paid',hue='credit.policy')
df['purpose'].nunique()
cat_feats=['purpose']
finaldf=pd.get_dummies(df,columns=cat_feats,drop_first=True)
finaldf.head()
from sklearn.model_selection import train_test_split
X = finaldf.drop(['not.fully.paid'],axis=1)
y =finaldf['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
predictions = dtc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
predicts=rfc.predict(X_test)
print(confusion_matrix(y_test,predicts))
print(classification_report(y_test,predicts))
