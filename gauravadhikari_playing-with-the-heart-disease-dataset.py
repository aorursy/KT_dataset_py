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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv("../input/heart.csv")
df.info()
df.head()
df.describe()
df.corr()
df.hist(figsize=(25,15))
plt.figure(figsize=(20,10))

sns.boxplot(data=df),
sns.countplot(data=df,x='target',hue='sex')

plt.title("Heart disease by gender")
X=df.drop('target',axis=1)

y=df['target']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaler.fit(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,predictions)
confusion_matrix(y_test,predictions)