import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline

df= pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
df.describe()
df['quality'].plot(kind='hist')
sns.boxplot(x='quality', y='pH',data=df)
correlation= df.corr()

plt.figure(figsize=(30,10))

sns.heatmap(correlation,annot=True)
plt.figure(figsize=(5,20))

sns.heatmap(correlation[['quality']].sort_values(by=['quality'], ascending=False), annot=True)
sns.regplot(x='quality', y='volatile acidity', data=df)
sns.regplot(x='quality', y='alcohol', data=df)
sns.regplot(x='quality', y='pH', data=df)
sns.regplot(x='quality', y='sulphates', data=df)
sns.regplot(x='quality', y='total sulfur dioxide', data=df)
sns.regplot(x='quality', y='citric acid', data=df)
bins= (2,6.5,8)

group_names=['bad','good']

df['quality']= pd.cut(df['quality'], bins= bins, labels= group_names)

df.head()

from sklearn.preprocessing import LabelEncoder

label= LabelEncoder()

df['quality']= label.fit_transform(df['quality'])
sns.countplot(df['quality'])
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
y= df['quality']

y.head()
x= df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol']]

x.head()

x.shape
train_x, test_x, train_y, test_y= train_test_split(x,y, test_size= 0.2, random_state=2)
Ctree= DecisionTreeClassifier(criterion='entropy')

Ctree.fit(train_x,train_y)
predtree= Ctree.predict(test_x)

print(predtree[:5])

print(test_y[:5])
from sklearn.metrics import accuracy_score

print("Decision tree accuracy: {}". format(accuracy_score(test_y, predtree)))
from sklearn.ensemble import RandomForestClassifier

model= RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')

model.fit(train_x,train_y)
pred_forest= model.predict(test_x)

pred_prob= model.predict_proba(test_x)[:,1]
print("Random Forest accuracy: {}". format(accuracy_score(test_y, pred_forest)))
from sklearn.svm import SVC

svc= SVC()

svc.fit(train_x, train_y)
pred_svc= svc.predict(test_x)

print("SVM accuracy: {}". format(accuracy_score(test_y, pred_svc)))