%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/athlete_events.csv')

df.head()
df.info()
df.isnull().sum()
df.dropna(inplace=True)
df.describe()
y = np.array([len(df[df.Sex =='F']),len(df[df.Sex =='M'])])

x = ['Female','Male']

plt.bar(x,y)
fig = plt.subplots(figsize=(10,5))

d = sns.countplot(x= 'Sport', data=df, hue='Season',palette='bright')

_ = plt.setp(d.get_xticklabels(),rotation = 90)
fig = plt.subplots(figsize=(10,5))

d = sns.countplot(x='Team' ,data=df.head(100), hue='Medal',palette='bright')

_ = plt.setp(d.get_xticklabels(),rotation=90)
plt.scatter(df.Height,df.Weight)

plt.xlabel('Heighy')

plt.ylabel('Weight')
fig = plt.subplots(figsize=(10,5))

d = sns.countplot(x='Age',data=df,hue='Medal',palette='bright')

_ = plt.setp(d.get_xticklabels(),rotation=90)
d = sns.countplot(x='Season',data=df , hue='Medal',palette='muted')
fig = plt.subplots(figsize=(10,5))

d = sns.countplot(x='City',data=df,hue='Season',palette='muted')

_ = plt.setp(d.get_xticklabels(),rotation=90)
fig = plt.subplots(figsize=(10,5))

d = sns.countplot(x='Year',data=df,hue='Season',palette='muted')

_ = plt.setp(d.get_xticklabels(),rotation=90)
df['Sex'] = df['Sex'].apply(lambda x: int(str(x).replace('F','1')) if 'F' in str(x) else x)

df['Sex'] = df['Sex'].apply(lambda x: int(str(x).replace('M','0')) if 'M' in str(x) else x)

df['Season'] = df['Season'].apply(lambda x: int(str(x).replace('Winter','1')) if 'Winter' in str(x) else x)

df['Season'] = df['Season'].apply(lambda x: int(str(x).replace('Summer','0')) if 'Summer' in str(x) else x)

df.info()
x = df[['Year','Height','Weight','Age','Sex']]

y = df['Season']
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score,recall_score

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)
lcr = LogisticRegression()

lcr.fit(x_train,y_train)

predict_lcr = lcr.predict(x_test)

print('Score:',lcr.score(x_test,y_test))

print('f1 score:',f1_score(y_test,predict_lcr,average='micro'))

print('precision score:',precision_score(y_test,predict_lcr,average='micro'))

print('recall score:',recall_score(y_test,predict_lcr,average='micro'))



cm_lcr = confusion_matrix(y_test,predict_lcr)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm_lcr,ax=ax,annot=True,linewidth=0.5,fmt='.0f')
svm = SVC()

svm.fit(x_train,y_train)

predict_svm = svm.predict(x_test)

print('Score:',svm.score(x_test,y_test))

print('f1 score:',f1_score(y_test,predict_svm,average='micro'))

print('precision score:',precision_score(y_test,predict_svm,average='micro'))

print('recall score:',recall_score(y_test,predict_svm,average='micro'))



cm_svm = confusion_matrix(y_test,predict_svm)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm_svm,ax=ax,annot=True,linewidth=0.5,fmt='.0f')
gnb = GaussianNB()

gnb.fit(x_train,y_train)

predict_gnb = gnb.predict(x_test)

print('Score:',gnb.score(x_test,y_test))

print('f1 score:',f1_score(y_test,predict_gnb,average='micro'))

print('precision score:',precision_score(y_test,predict_gnb,average='micro'))

print('recall score:',recall_score(y_test,predict_gnb,average='micro'))



cm_gnb = confusion_matrix(y_test,predict_gnb)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm_gnb,ax=ax,annot=True,linewidth=0.5,fmt='.0f')
dtc = DecisionTreeClassifier()

dtc.fit(x_train,y_train)

predict_dtc = dtc.predict(x_test)

print('Score:',dtc.score(x_test,y_test))

print('f1 score:',f1_score(y_test,predict_dtc,average='micro'))

print('precision score:',precision_score(y_test,predict_dtc,average='micro'))

print('recall score:',recall_score(y_test,predict_dtc,average='micro'))



cm_dtc = confusion_matrix(y_test,predict_dtc)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm_dtc,ax=ax,annot=True,linewidth=0.5,fmt='.0f')
rfc = RandomForestClassifier(n_estimators=100,random_state=10)

rfc.fit(x_train,y_train)

predict_rfc = rfc.predict(x_test)

print('Score:',rfc.score(x_test,y_test))

print('f1 score:',f1_score(y_test,predict_rfc,average='micro'))

print('precision score:',precision_score(y_test,predict_rfc,average='micro'))

print('recall score:',recall_score(y_test,predict_rfc,average='micro'))



cm_rfc = confusion_matrix(y_test,predict_rfc)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm_rfc,ax=ax,annot=True,linewidth=0.5,fmt='.0f')