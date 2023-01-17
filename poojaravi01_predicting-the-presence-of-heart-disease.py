import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
df=pd.read_csv('../input/heart-disease-uci/heart.csv')

df.head()
df.shape
df.isna().sum()
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(),annot=True,linewidth=0.2,cmap='coolwarm')
plt.figure(figsize=(20,15))

plt.subplot(4,4,1)

sns.countplot(data=df,x='sex',hue='target',palette='Set2')

plt.subplot(4,4,2)

sns.countplot(data=df,x='cp',hue='target',palette='Set2')

plt.subplot(4,4,3)

sns.countplot(data=df,x='fbs',hue='target',palette='Set2')

plt.subplot(4,4,4)

sns.countplot(data=df,x='restecg',hue='target',palette='Set2')

plt.subplot(4,4,5)

sns.countplot(data=df,x='exang',hue='target',palette='Set1')

plt.subplot(4,4,6)

sns.countplot(data=df,x='slope',hue='target',palette='Set1')

plt.subplot(4,4,7)

sns.countplot(data=df,x='ca',hue='target',palette='Set1')

plt.subplot(4,4,8)

sns.countplot(data=df,x='thal',hue='target',palette='Set1')
plt.figure(figsize=(20,15))

plt.subplot(4,4,1)

sns.distplot(a=df['age'],bins=30)

plt.subplot(4,4,2)

sns.distplot(a=df['trestbps'],bins=40,color='red')

plt.subplot(4,4,3)

sns.distplot(a=df['chol'],bins=50,color='green')

plt.subplot(4,4,4)

sns.distplot(a=df['oldpeak'],bins=30,color='purple')
g = sns.catplot(x="age", y="target", row="sex",kind="box", orient="h", height=1.5, aspect=4,data=df,palette='Set2')

g.set(xscale='log')
g = sns.catplot(x="trestbps", y="target", row="sex",kind="box", orient="h", height=1.5, aspect=4,data=df,palette='Set2')

g.set(xscale='log')
g = sns.catplot(x="chol", y="target", row="sex",kind="box", orient="h", height=1.5, aspect=4,data=df,palette='Set2')

g.set(xscale='log')
g = sns.catplot(x="oldpeak", y="target", row="sex",kind="box", orient="h", height=1.5, aspect=4,data=df,palette='Set2')
d1=pd.get_dummies(df['cp'],drop_first=True,prefix='cp')

d2=pd.get_dummies(df['thal'],drop_first=True,prefix='thal')

d3=pd.get_dummies(df['slope'],drop_first=True,prefix='slope')

df=pd.concat([df,d1,d2,d3],axis=1)

df.drop(['cp','thal','slope'],axis=1,inplace=True)

df.head()
df['age'].min()
df['age'].max()
df['seniors'] = df['age'].map(lambda s: 1 if s >= 60 else 0)
df.head()
X=df.drop('target',axis=1)

y=df['target']
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)
scale=StandardScaler()

xtrain=scale.fit_transform(xtrain)

xtest=scale.transform(xtest)
scores=[]
clf1=LogisticRegression()

clf1.fit(xtrain,ytrain)

pred1=clf1.predict(xtest)

s1=accuracy_score(ytest,pred1)

scores.append(s1*100)

print(s1*100)
clf2=RandomForestClassifier(max_depth=2,random_state=0)

clf2.fit(xtrain,ytrain)

pred2=clf2.predict(xtest)

s2=accuracy_score(ytest,pred2)

scores.append(s2*100)

print(s2*100)
clf3=KNeighborsClassifier()

clf3.fit(xtrain,ytrain)

pred3=clf3.predict(xtest)

s3=accuracy_score(ytest,pred3)

scores.append(s3*100)

print(s3*100)
clf4=svm.SVC(kernel='rbf',C=1)

clf4.fit(xtrain,ytrain)

pred4=clf4.predict(xtest)

s4=accuracy_score(ytest,pred4)

scores.append(s4*100)

print(s4*100)
clf5=DecisionTreeClassifier(max_depth=3,random_state=0)

clf5.fit(xtrain,ytrain)

pred5=clf5.predict(xtest)

s5=accuracy_score(ytest,pred5)

scores.append(s5*100)

print(s5*100)
print(scores)
names=['LogisticRegression','RandomForest','KNN','SVM','Decision Tree']

classifier=pd.Series(data=scores,index=names)

print(classifier)
plt.figure(figsize=(10,7))

classifier.sort_index().plot.bar()
print(confusion_matrix(ytest,pred1))
print(classification_report(ytest,pred1))