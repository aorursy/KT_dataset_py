# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/kyphosis-dataset/kyphosis.csv')

df.tail()
df.info()
sns.pairplot(df,hue='Kyphosis',palette='Set1');
from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Kyphosis', axis=1),df['Kyphosis'],test_size=0.30)
from  sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
prediction = dtree.predict(X_test)
from  sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test, prediction)
print(classification_report(y_test, prediction))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
confusion_matrix(y_test,rfc_pred)
df['Kyphosis'].value_counts()