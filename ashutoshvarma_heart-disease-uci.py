# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()
df.info()
df.isnull().sum()
sns.boxplot(x='target',y='age',data=df)
sns.catplot(x='target',y='age',hue='cp',data=df)
sns.countplot(x='target',hue='sex',data=df,color='green')
X = df.drop(columns='target')
X.head()
y = df['target']
y.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.3)
from sklearn.linear_model import LogisticRegression

linear = LogisticRegression()

linear.fit(X_train,y_train)
y_pred = linear.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))