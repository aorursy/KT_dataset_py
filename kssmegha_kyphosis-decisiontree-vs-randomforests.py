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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/kyphosis.csv')

df.head()
df.info()
sns.pairplot(df,hue='Kyphosis')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
X=df.drop('Kyphosis',axis=1)

y=df['Kyphosis']
from sklearn.tree import DecisionTreeClassifier

dtree =  DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dp = predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(predictions,y_test))

print('/n')

print(classification_report(predictions,y_test))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(rfc_pred,y_test))

print('/n')

print(classification_report(rfc_pred,y_test))
# Random forests with accuracy of 78%

print(rfc_pred)
# Decision tree with accuracy of 59%

dp