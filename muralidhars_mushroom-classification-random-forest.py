# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
df.keys()
df.info()
df.shape

df.isnull().sum()
from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()

df.iloc[:,0]=le.fit_transform(df.iloc[:,0])
df.head()
df=(df.apply(LabelEncoder().fit_transform))
df.head()
x=df.drop(['class'],axis=1)

y=df['class']
x.head()
y.head()
plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot=True)
print(df.isnull().sum())
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=10)
print(x_train.shape,y_train.shape)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)

print(cm)