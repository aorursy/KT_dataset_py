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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("../input/heart.csv")
df.shape
df.describe()
df.isnull().values.any()
plt.hist(df['target'])
plt.hist(df['age'])
plt.hist(df['sex'])
X = df[['age','sex','cp','trestbps', 'chol', 'fbs','restecg','thalach','exang','oldpeak','ca']]

y = df.iloc[:,-1]
f, ax = plt.subplots(figsize=(10,8))

corr = X.corr()

sns.heatmap(corr,

           xticklabels = corr.columns.values,

           yticklabels = corr.columns.values)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 42)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr = lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
y_pred = lr.predict([[63,1,3,145,223,1,0,150,0,2.3,0]])

y_pred