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
df = pd.read_csv('../input/data.csv')

df.head()
df.shape
df.info()
df = df.drop('Unnamed: 32', axis='columns')
df['diagnosis'].unique()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['diagnosis'] = le.fit_transform(df['diagnosis'])

df.tail()
y = df['diagnosis']

X = df.drop(['diagnosis','id'],axis='columns')
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X,y,test_size=0.2)
X_train.shape
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train,y_train)

model.score(X_test,y_test)
from sklearn.model_selection import cross_val_score
score1 = cross_val_score(RandomForestClassifier(),X,y)

np.average(score1)
model = RandomForestClassifier(n_estimators=50,criterion='gini')

model.fit(X_train,y_train)

model.score(X_test,y_test)
scores2 = cross_val_score(model,X,y,cv=10)

np.average(scores2)
from sklearn.svm import SVC

model = SVC()
scores3 =  cross_val_score(model,X,y,cv=10)

np.average(scores3)
model = SVC(gamma=0.01)

scores3 = cross_val_score(model,X,y,cv=10)

np.average(scores3)
from sklearn.ensemble import BaggingClassifier

BA = BaggingClassifier(n_estimators=100)

BA_model = BA.fit(X_train, y_train)