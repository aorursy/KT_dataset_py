import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import time
df = pd.read_csv("../input/heart-dataset/heart_info.csv")
df.shape
df.info()
df.head()
#for target

conversion_dict = {1 : 'isHeartPatient', 0 : 'isNotHeartPatient'}

df['target'] = df['target'].replace(conversion_dict)

#for sex

conversion_dict = {1 : 'Male', 0 : 'Female'}

df['sex'] = df['sex'].replace(conversion_dict)

#for cp

conversion_dict = {0 : 'Typical', 1 : 'Atypical', 2 : 'Non-anginal', 3 : 'Asymptomatic'}

df['cp'] = df['cp'].replace(conversion_dict)

#for fbs

conversion_dict = {1 : 'fbs > 120 mg/dl', 0 : 'fbs < 120 mg/dl'}

df['fbs'] = df['fbs'].replace(conversion_dict)

#for exang

conversion_dict = {1 : 'induced angina', 0 : 'not induced angina'}

df['exang'] = df['exang'].replace(conversion_dict)

df.head()
df.isnull().sum()
df_plot = df.groupby(['target', 'sex']).size().reset_index().pivot(columns='target', index='sex', values=0)

df_plot.plot(kind='bar', stacked=True, color=['skyblue','orange'])
df_plot = df.groupby(['target', 'cp']).size().reset_index().pivot(columns='target', index='cp', values=0)

df_plot.plot(kind='bar', stacked=True, color=['yellowgreen','violet'])
df_plot = df.groupby(['target', 'fbs']).size().reset_index().pivot(columns='target', index='fbs', values=0)

df_plot.plot(kind='bar', stacked=True, color=['orange', 'yellowgreen'])
df_plot = df.groupby(['target', 'exang']).size().reset_index().pivot(columns='target', index='exang', values=0)

df_plot.plot(kind='bar', stacked=True, color=['gold', 'violet'])
sns.distplot(df['thalach'],kde=True,bins=30,color='green')
sns.distplot(df['chol'],kde=True,bins=30,color='red')
sns.distplot(df['trestbps'],kde=True,bins=30,color='blue')
plt.figure(figsize=(15,6))

sns.countplot(x='age',data = df, hue = 'target',palette='cubehelix')
#for target

conversion_dict = {'isHeartPatient' : 1, 'isNotHeartPatient' : 0}

df['target'] = df['target'].replace(conversion_dict)

#for sex

conversion_dict = {'Male' : 1,'Female' : 0}

df['sex'] = df['sex'].replace(conversion_dict)

#for cp

conversion_dict = {'Typical' : 0,'Atypical' : 1,'Non-anginal' : 2,'Asymptomatic' : 3}

df['cp'] = df['cp'].replace(conversion_dict)

#for fbs

conversion_dict = {'fbs > 120 mg/dl' : 1, 'fbs < 120 mg/dl' : 0}

df['fbs'] = df['fbs'].replace(conversion_dict)

#for exang

conversion_dict = {'induced angina' : 1,'not induced angina' : 0}

df['exang'] = df['exang'].replace(conversion_dict)

df.head()
x = df.drop('target',axis=1)

y = df['target']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=42) 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x_train)



x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
begin = time.time()

y_pred = logreg.predict(x_test)

end = time.time()

lrExecTime = end - begin

print('Execution Time taken by LR : ',lrExecTime)
y_pred
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))

print('Accuracy Score: ',accuracy_score(y_test,y_pred))

lrAccuracy = round(accuracy_score(y_test,y_pred),5)*100

print('Using Logistic Regression we get an accuracy score of: ',

      lrAccuracy,'%')
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
begin = time.time()

y_pred = gnb.predict(x_test)

end = time.time()

gnbExecTime = end - begin

print('Execution Time taken by GNB : ',gnbExecTime)
y_pred
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))

print('Accuracy Score: ',accuracy_score(y_test,y_pred))

gnbAccuracy = round(accuracy_score(y_test,y_pred),5)*100

print('Using Gaussian Naive Bayesian we get an accuracy score of: ',

      gnbAccuracy,'%')
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(x_train, y_train)
begin = time.time()

y_pred = classifier.predict(x_test)

end = time.time()

knnExecTime = end - begin

print('Execution Time taken by KNN : ',knnExecTime)
y_pred
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy Score: ',accuracy_score(y_test,y_pred))

knnAccuracy = round(accuracy_score(y_test,y_pred),5)*100

print('Using k-NN we get an accuracy score of: ',

      knnAccuracy,'%')
labels = 'LR','GNB','KNN'

values = [lrExecTime,gnbExecTime,knnExecTime]

plt.bar(labels,values,color=['violet','skyblue','yellowgreen'])

plt.title('Execution Time Comparison')

plt.show()
labels = 'LR','GNB','KNN'

values = [lrAccuracy,gnbAccuracy,knnAccuracy]

plt.bar(labels,values,color=['skyblue','gold','violet'])

plt.title('Accuracy Comparison')

plt.show()