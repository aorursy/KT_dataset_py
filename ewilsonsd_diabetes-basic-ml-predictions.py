import numpy as np 

import pandas as pd 

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



warnings.filterwarnings("ignore")



%matplotlib inline
df = pd.read_csv('../input/diabetes-dataset/diabetes.csv')
df.head()
corr = df.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
corr[abs(corr['Outcome']) > 0.2]['Outcome']
corr[abs(corr['Outcome']) > 0.1]['Outcome']
small_df=df[['Pregnancies', 'Glucose', 'BMI', 'Age']]

med_df=df[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

large_df = df.drop(columns=['Outcome'])
xs = small_df

xm = med_df

xl = large_df

y = df['Outcome']
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr=LogisticRegression(max_iter=10000)
x_train,x_test,y_train,y_test=train_test_split(xs,y,random_state=1,test_size=0.2)

lr.fit(x_train,y_train)

p1=lr.predict(x_test)

s1=accuracy_score(y_test,p1)

print("Small DF Linear Regression Success Rate :", "{:.2f}%".format(100*s1))
x_train,x_test,y_train,y_test=train_test_split(xm,y,random_state=1,test_size=0.2)

lr.fit(x_train,y_train)

p1=lr.predict(x_test)

s1=accuracy_score(y_test,p1)

print("Medium DF Linear Regression Success Rate :", "{:.2f}%".format(100*s1))
x_train,x_test,y_train,y_test=train_test_split(xl,y,random_state=1,test_size=0.2)

lr.fit(x_train,y_train)

p1=lr.predict(x_test)

s1=accuracy_score(y_test,p1)

print("Large DF Linear Regression Success Rate :", "{:.2f}%".format(100*s1))
from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier()
x_train,x_test,y_train,y_test=train_test_split(xs,y,random_state=1,test_size=0.2)

gbc.fit(x_train,y_train)

p2=gbc.predict(x_test)

s2=accuracy_score(y_test,p2)

print("Small DF Gradient Booster Classifier Success Rate :", "{:.2f}%".format(100*s2))
x_train,x_test,y_train,y_test=train_test_split(xm,y,random_state=1,test_size=0.2)

gbc.fit(x_train,y_train)

p2=gbc.predict(x_test)

s2=accuracy_score(y_test,p2)

print("Medium DF Gradient Booster Classifier Success Rate :", "{:.2f}%".format(100*s2))
x_train,x_test,y_train,y_test=train_test_split(xl,y,random_state=1,test_size=0.2)

gbc.fit(x_train,y_train)

p2=gbc.predict(x_test)

s2=accuracy_score(y_test,p2)

print("Large DF Gradient Booster Classifier Success Rate :", "{:.2f}%".format(100*s2))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
x_train,x_test,y_train,y_test=train_test_split(xs,y,random_state=1,test_size=0.2)

rfc.fit(x_train,y_train)

p3=rfc.predict(x_test)

s3=accuracy_score(y_test,p3)

print("Small DF Random Forest Classifier Success Rate :", "{:.2f}%".format(100*s3))
x_train,x_test,y_train,y_test=train_test_split(xm,y,random_state=1,test_size=0.2)

rfc.fit(x_train,y_train)

p3=rfc.predict(x_test)

s3=accuracy_score(y_test,p3)

print("Medium DF Random Forest Classifier Success Rate :", "{:.2f}%".format(100*s3))
x_train,x_test,y_train,y_test=train_test_split(xl,y,random_state=1,test_size=0.2)

rfc.fit(x_train,y_train)

p3=rfc.predict(x_test)

s3=accuracy_score(y_test,p3)

print("Large DF Random Forest Classifier Success Rate :", "{:.2f}%".format(100*s3))
from sklearn.svm import SVC

svm=SVC()
x_train,x_test,y_train,y_test=train_test_split(xs,y,random_state=1,test_size=0.2)

svm.fit(x_train,y_train)

p4=svm.predict(x_test)

s4=accuracy_score(y_test,p4)

print("Small DF Support Vector Classifier Success Rate :", "{:.2f}%".format(100*s4))
x_train,x_test,y_train,y_test=train_test_split(xm,y,random_state=1,test_size=0.2)

svm.fit(x_train,y_train)

p4=svm.predict(x_test)

s4=accuracy_score(y_test,p4)

print("Medium DF Support Vector Classifier Success Rate :", "{:.2f}%".format(100*s4))
x_train,x_test,y_train,y_test=train_test_split(xl,y,random_state=1,test_size=0.2)

svm.fit(x_train,y_train)

p4=svm.predict(x_test)

s4=accuracy_score(y_test,p4)

print("Large DF Support Vector Classifier Success Rate :", "{:.2f}%".format(100*s4))