import pandas as pd 

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt
df=pd.read_csv('../input/heart-disease-uci/heart.csv')

df.head()
sns.heatmap(df.corr())
sns.countplot(df['cp'])
sns.pairplot(df)
sns.countplot(x='target',data=df)
df.isnull().sum().max()
dummy1=pd.get_dummies(df.cp)

dummy2=pd.get_dummies(df.restecg)

dummy3=pd.get_dummies(df.slope)

dummy4=pd.get_dummies(df.ca)

dummy5=pd.get_dummies(df.thal)

total=pd.concat([df,dummy1,dummy2,dummy3,dummy4,dummy5],axis='columns')
df=total.drop(['cp','restecg','slope','ca','thal'],axis=1)
df.head()
x=df.drop('target',axis=1)

y=df['target']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(random_state = 0)

model1.fit(x_train, y_train)
y_pred1= model1.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cm = confusion_matrix(y_test, y_pred1)

print(cm)

accuracy_score(y_test, y_pred1)

print(classification_report(y_test,y_pred1))
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
n_estimators=[100,250,500,1000]

criterion=['entropy','gini']

max_features = ['auto','sqrt','log2']

parameters={'n_estimators':[100,250,500,1000],'criterion':['entropy','gini'],'max_features' : ['auto','sqrt','log2']}

rf=RandomForestClassifier()

gridsearch=GridSearchCV(rf,parameters,scoring='neg_mean_squared_error',cv=5)

gridsearch.fit(x_train,y_train)

gridsearch.best_params_
model2 = RandomForestClassifier(n_estimators=100,criterion= 'entropy',max_features = 'log2',random_state=5)

model2.fit(x_train, y_train)

y_pred2=model2.predict(x_test)

cm=confusion_matrix(y_test,y_pred2)

print(cm)

accuracy_score(y_test,y_pred2)
print(classification_report(y_test,y_pred2))