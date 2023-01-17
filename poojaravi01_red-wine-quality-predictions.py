from sklearn import svm

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error
df=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
df.isna().sum()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.distplot(df['alcohol'],bins=50)

plt.subplot(2,2,2)

sns.distplot(df['pH'],bins=50,color='red')

plt.subplot(2,2,3)

sns.distplot(df['sulphates'],bins=50,color='green')

plt.subplot(2,2,4)

sns.distplot(df['total sulfur dioxide'],bins=50,color='purple')
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.distplot(df['fixed acidity'],bins=50)

plt.subplot(2,2,2)

sns.distplot(df['volatile acidity'],bins=50,color='red')

plt.subplot(2,2,3)

sns.distplot(df['citric acid'],bins=50,color='green')

plt.subplot(2,2,4)

sns.distplot(df['residual sugar'],bins=50,color='purple')
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.distplot(df['chlorides'],bins=50)

plt.subplot(2,2,2)

sns.distplot(df['free sulfur dioxide'],bins=50,color='red')

plt.subplot(2,2,3)

sns.distplot(df['density'],bins=50,color='green')
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(),annot=True,linewidth=0.2)
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.boxplot(df['quality'],df['alcohol'])

plt.subplot(2,2,2)

sns.boxplot(df['quality'],df['sulphates'])

plt.subplot(2,2,3)

sns.boxplot(df['quality'],df['pH'])

plt.subplot(2,2,4)

sns.boxplot(df['quality'],df['density'])
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.boxplot(df['quality'],df['fixed acidity'])

plt.subplot(2,2,2)

sns.boxplot(df['quality'],df['volatile acidity'])

plt.subplot(2,2,3)

sns.boxplot(df['quality'],df['residual sugar'])

plt.subplot(2,2,4)

sns.boxplot(df['quality'],df['citric acid'])
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.boxplot(df['quality'],df['free sulfur dioxide'])

plt.subplot(2,2,2)

sns.boxplot(df['quality'],df['total sulfur dioxide'])
df['quality'].min()
df['quality'].max()
values = (2, 6, 9)

qual = ['bad', 'good']

df['quality'] = pd.cut(df['quality'], bins = values, labels = qual)

df.head()
df['quality'].value_counts()
le=LabelEncoder()

df['quality']=le.fit_transform(df['quality'])
X=df.drop('quality',axis=1)

y=df['quality']
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()

xtrain=sc.fit_transform(xtrain)

xtest=sc.fit_transform(xtest)
model=svm.SVC()

model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print(accuracy_score(ytest,ypred))
rf=RandomForestClassifier()

rf.fit(xtrain,ytrain)

y0pred=rf.predict(xtest)

print(accuracy_score(ytest,y0pred))
xgb=XGBClassifier(max_depth=3,n_estimators=200,learning_rate=0.5)

xgb.fit(xtrain,ytrain)

y1pred=xgb.predict(xtest)

print(accuracy_score(ytest,y1pred))
print(confusion_matrix(ytest,y1pred))
print(classification_report(ytest,y1pred))