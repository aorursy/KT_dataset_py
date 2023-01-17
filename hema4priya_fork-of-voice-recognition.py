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
df='../input/voice.csv'

d=pd.read_csv(df)

print(d)
print(d.info())
print(d.head(10))
print(d.describe())
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
x = np.array(d.drop(['label'], 1))

y = np.array(d['label'])
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df=StandardScaler()

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

X_train=df.fit(X_train)

X_test=df.fit(X_test)
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model = LogisticRegression()

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print('Accuracy='+str(accuracy))
sns.heatmap(d.corr(),linewidths=0.25,vmax=1.0, square=True)