import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt

from patsy import dmatrices

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

import seaborn as sb

import csv

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

import pandas.tseries

%matplotlib inline

from pylab import *
df2=pd.read_csv("../input/Corporate.csv")

df2.head()
d,C=dmatrices('Hired ~ PERCENTAGE + BACKLOG + INTERNSHIP + FIRSTROUND + COMMUNICATIONSKILLLS', df2, return_type='dataframe')

A=df2[['PERCENTAGE','BACKLOG','INTERNSHIP','FIRSTROUND','COMMUNICATIONSKILLLS']]

b=df2[['Hired']]

feat=list(df2.columns[1:6])

feat

X=df2[feat]

y=df2['Hired']
A_train, A_test, b_train, b_test = train_test_split(A,b,test_size=0.25,random_state=23)

model1=SVC(kernel='linear')

model1=model1.fit(A_train, b_train)

C_train, C_test, d_train, d_test = train_test_split(C,d,test_size=0.25,random_state=27)

model2=LogisticRegression()

model2=model2.fit(C_train, d_train)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=29)

model3=tree.DecisionTreeClassifier()

model3=model3.fit(X_train, y_train)

model4=RandomForestClassifier(n_estimators=40)

model4=model4.fit(X_train, y_train)
df2.info()
# n=input('Name:- ')

# a=int(input('Percentage :- '))

# b=int(input('No. of Backlogs :- '))

# c=int(input('No. of Internships :- '))

# d=int(input('Technical Round Score :- '))

# e=int(input('Comm. Skills Score :- '))

n='User Name'

a=78

b=2

c=1

d=81

e=79
my_val=np.array([a,b,c,d,e]).reshape(1,-1)

g=model1.predict(my_val)

h=model1.score(A_test,b_test)

if g[0]==0:

    r="NO"

elif g[0]==1:

    r="YES"

print('Support Vector Classifier -->')

print('Result :- '+r)

print('Score :- '+str(h))
my_val=np.array([1,a,b,c,d,e]).reshape(1,-1)

g=model2.predict(my_val)

h=model2.score(C_test,d_test)

if g[0].astype(int)==0:

    r="NO"

elif g[0].astype(int)==1:

    r="YES"

print('Logistic Regression -->')

print('Result :- '+r)

print('Score :- '+str(h))
my_val=np.array([a,b,c,d,e]).reshape(1,-1)

g=model3.predict(my_val)

h=model3.score(X_test,y_test)

if g[0]==0:

    r="NO"        

elif g[0]==1:

    r="YES"

print('Decision Tree -->')

print('Result :- '+r)

print('Score :- '+str(h))
my_val=np.array([a,b,c,d,e]).reshape(1,-1)

g=model4.predict(my_val)

h=model4.score(X_test,y_test)

if g[0]==0:

    r="NO"

elif g[0]==1:

    r="YES"

print('Random Forest -->')

print('Result :- '+r)

print('Score :- '+str(h))
my_val=np.array([1,a,b,c,d,e]).reshape(1,-1)

my_val1=np.array([a,b,c,d,e]).reshape(1,-1)

i=model1.predict(my_val1)

j=model2.predict(my_val)

k=model3.predict(my_val1)

l=model4.predict(my_val1)

w=i[0]

x=j[0].astype(int)

y=k[0]

z=l[0]

sum=w+x+y+z

if sum<2:

    op="NO"

else:

    op="YES"

print('Optimised Algorithm -->')

print('Result :- '+op)



field=[n,a,b,c,d,e,op]

with open('Corporate.csv','a',newline='') as p:

    writer=csv.writer(p)

    writer.writerow(field)

    