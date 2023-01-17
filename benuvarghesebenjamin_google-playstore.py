import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
data=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
data.head()
data.shape
data.nunique()
data.isnull().sum()
outlier=data[data['Rating']>5]

outlier
data.drop(10472,inplace=True)
data.hist(bins=40,color='fuchsia')

plt.show()
plt.figure(figsize=(5,5))

data['Type'].value_counts().plot.bar(color='black')

plt.show()
plt.rcParams['figure.figsize']=(12,6)
data['Category'].value_counts().plot.bar()

plt.show()
data['Content Rating'].value_counts().plot.bar(color='gray')
plt.figure(figsize=(20,20))

sns.barplot(x='Installs',y='Rating',data=data)

plt.show()
plt.rcParams['figure.figsize']=(25,12)

sns.barplot(x='Android Ver',y='Rating',data=data)

plt.show()
category=pd.get_dummies(data['Category'],drop_first=True)

types=pd.get_dummies(data['Type'],drop_first=True)

content=pd.get_dummies(data['Content Rating'],drop_first=True)

new=[data,category,types]

data=pd.concat(new,axis=1)

data.drop(['Category','Installs','Type','Content Rating'],axis=1,inplace=True)
data.head()
data.drop(['App','Size','Price','Genres','Last Updated','Current Ver','Android Ver'],axis=1,inplace=True)
data.head()
X=data.drop('Rating',axis=1)

y=data['Rating'].values

y=y.astype('int')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=102)
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_train=ss.fit_transform(X_train)

X_test=ss.transform(X_test)
from sklearn.linear_model import LogisticRegression
logr=LogisticRegression()
model=logr.fit(X_train,y_train)
prediction=model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction)
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(random_state=123,criterion='entropy')
model=dtc.fit(X_train,y_train)
prediction=model.predict(X_test)
accuracy_score(y_test,prediction)
confusion_matrix(y_test,prediction)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=456)
model=rfc.fit(X_train,y_train)
prediction=model.predict(X_test)
accuracy_score(y_test,prediction)
confusion_matrix(y_test,prediction)