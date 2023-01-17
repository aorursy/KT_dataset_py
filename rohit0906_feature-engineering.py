# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/weight-height/weight-height.csv')
df.head()
df.describe()
import matplotlib.pyplot as plt
plt.hist(df.Height, bins=20, rwidth=0.8)

plt.xlabel('Height (inches)')

plt.ylabel('Count')

plt.show()

plt.hist(df.Weight, bins=20, rwidth=0.8)

plt.xlabel('Weight (inches)')

plt.ylabel('Count')

plt.show()
df_male=df[(df['Gender']=='Male')]

df_male.shape
df_female=df[(df['Gender']=='Female')]

df_female.shape
Q3=df_male.Height.quantile(0.75)

Q1=df_male.Height.quantile(0.25)

max_threshold=Q3+1.5*(Q3-Q1)

min_threshold=Q1-1.5*(Q3-Q1)

df_male=df_male[(df_male['Height']<=max_threshold) &( df_male['Height']>=min_threshold)]

Q3=df_male.Weight.quantile(0.75)

Q1=df_male.Weight.quantile(0.25)

max_threshold=Q3+1.5*(Q3-Q1)

min_threshold=Q1-1.5*(Q3-Q1)

df_male=df_male[(df_male['Weight']<=max_threshold) &( df_male['Weight']>=min_threshold)]

df_male.shape
Q3=df_female.Height.quantile(0.75)

Q1=df_female.Height.quantile(0.25)

max_threshold=Q3+1.5*(Q3-Q1)

min_threshold=Q1-1.5*(Q3-Q1)

df_female=df_female[(df_female['Height']<=max_threshold) &( df_female['Height']>=min_threshold)]

Q3=df_female.Weight.quantile(0.75)

Q1=df_female.Weight.quantile(0.25)

max_threshold=Q3+1.5*(Q3-Q1)

min_threshold=Q1-1.5*(Q3-Q1)

df_female=df_female[(df_female['Weight']<=max_threshold) &( df_female['Weight']>=min_threshold)]

df_female.shape
df=df_male.append(df_female)

df.shape
from sklearn.utils import shuffle

df=shuffle(df)

df.head()
X=df.drop(labels=['Gender'], axis=1)

y=df['Gender']
X.shape
y.shape
from sklearn.preprocessing import LabelBinarizer

lb=LabelBinarizer()

y=lb.fit_transform(y)

y
plt.scatter(X.Height, X.Weight,y+1, c=y)

plt.show()
'''from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X=ss.fit_transform(X)

X'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25)
X_train.shape
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,y_train)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

pred=knn.predict(X_test)

score=accuracy_score(y_true=y_test, y_pred=pred)

print(score)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train,y_train)
pred=lr.predict(X_test)

score=accuracy_score(y_true=y_test, y_pred=pred)

print(score)
from sklearn.svm import SVC

svc=SVC()

svc.fit(X_train,y_train)

pred=svc.predict(X_test)

score=accuracy_score(y_true=y_test, y_pred=pred)

print(score)
from sklearn.tree import DecisionTreeClassifier

tree=DecisionTreeClassifier()

tree.fit(X_train,y_train)

pred=tree.predict(X_test)

score=accuracy_score(y_true=y_test, y_pred=pred)

print(score)
matrix=confusion_matrix(y_true=y_test, y_pred=pred)

matrix
f1_score(y_true=y_test, y_pred=pred)
pred=lr.predict(X)

plt.scatter(X.Height, X.Weight,pred+1, c=pred)

plt.show()