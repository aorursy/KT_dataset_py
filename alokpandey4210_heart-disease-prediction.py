# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.tail()
df['target'].value_counts()
df.corr(method='pearson')
X=df.drop(['fbs','target'],axis=1)
y=df['target']
X.isnull().sum()
X=pd.get_dummies(X)
X['age'].hist(bins=50)
X.boxplot(column='age')
X['age'].plot('density')
X['sex'].hist(bins=50)
X.boxplot(column='sex')
X['sex'].plot('density')
X['cp'].hist(bins=50)
X.boxplot(column='cp')
X['cp'].plot('density')
X['trestbps'].hist(bins=60)
X.boxplot(column='trestbps')
X['trestbps'].plot('density')
X['chol'].hist(bins=50)
X.boxplot(column='chol')
X['chol'].plot('density',color='Red')
X['restecg'].hist(bins=60)
X['restecg'].plot('density')
X.boxplot(column='restecg')
X['thalach'].hist(bins=50)
X.boxplot(column='thalach')
X['thalach'].plot('density',color='Green')
X['exang'].hist(bins=50)
X.boxplot(column='exang')
X['exang'].plot('density',color='Yellow')
X['oldpeak'].hist(bins=50)
X['slope'].hist(bins=50)
X['ca'].hist(bins=50)
X['ca'].plot('density')
X['thal'].hist(bins=50)
X.boxplot(column='thal')
X['thal'].plot('density',color='Blue')
X.isnull().sum()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
from sklearn.linear_model import LogisticRegression

qs=LogisticRegression()

qs.fit(X_train,y_train)

qs.score(X_test,y_test)
from sklearn.tree import DecisionTreeRegressor

wx=DecisionTreeRegressor(max_depth=3)

wx.fit(X_train,y_train)

wx.score(X_test,y_test)
from sklearn.ensemble import RandomForestRegressor

oh=RandomForestRegressor()

oh.fit(X_train,y_train)

oh.score(X_test,y_test)
from sklearn.linear_model import LinearRegression

qc=LinearRegression()

qc.fit(X_train,y_train)

qc.score(X_test,y_test)
y_test.value_counts()
from sklearn.ensemble import RandomForestClassifier

pa=RandomForestClassifier()

pa.fit(X_train,y_train)

pa.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier

qm=DecisionTreeClassifier()

qm.fit(X_train,y_train)

qm.score(X_test,y_test)
from sklearn.svm import SVC

os=SVC()

os.fit(X_train,y_train)

os.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

ql=KNeighborsClassifier()

ql.fit(X_train,y_train)

ql.score(X_test,y_test)
#save model

import pickle 

file_name='DISEASE.sav'



tuples=(pa,X)

pickle.dump(tuples,open(file_name,'wb'))
from sklearn.metrics import confusion_matrix

ypp=qs.predict(X_test)

result1=confusion_matrix(y_test,ypp)

print(result1)
from sklearn.metrics import confusion_matrix

ypp1=pa.predict(X_test)

result2=confusion_matrix(y_test,ypp1)

print(result2)
from sklearn.metrics import confusion_matrix

ypp2=qm.predict(X_test)

result3=confusion_matrix(y_test,ypp2)

print(result3)
from sklearn.metrics import confusion_matrix

ypp3=os.predict(X_test)

result4=confusion_matrix(y_test,ypp3)

print(result4)
from sklearn.metrics import confusion_matrix

ypp4=ql.predict(X_test)

result5=confusion_matrix(y_test,ypp4)

print(result5)
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr,tpr,threshold=roc_curve(y_test,qs.predict_proba(X_test)[:,1])
fpr
tpr
threshold
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr1,tpr1,threshold1=roc_curve(y_test,pa.predict_proba(X_test)[:,1])
fpr1
tpr1
threshold1
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr2,tpr2,threshold2=roc_curve(y_test,qm.predict_proba(X_test)[:,1])
fpr2
tpr2
threshold2
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr3,tpr3,threshold3=roc_curve(y_test,ql.predict_proba(X_test)[:,1])
fpr3
tpr3
threshold3
roc_auc=roc_auc_score(y_test,qs.predict(X_test))

roc_auc1=roc_auc_score(y_test,pa.predict(X_test))



roc_auc3=roc_auc_score(y_test,os.predict(X_test))

roc_auc4=roc_auc_score(y_test,ql.predict(X_test))

print(roc_auc,roc_auc1,roc_auc3,roc_auc4)
y_prob=qs.predict_proba(X_test)
from sklearn.preprocessing import binarize

yBIN=binarize(y_prob,0.60)

yBIN
y_BIN1=yBIN[:,1]
yTYPE=y_BIN1.astype(int)
from sklearn.metrics import confusion_matrix

qq=confusion_matrix(y_test,yTYPE)

qq
from sklearn.preprocessing import binarize

yBIN=binarize(y_prob,0.70)

yBIN
y_BIN1=yBIN[:,1]
yTYPE=y_BIN1.astype(int)
from sklearn.metrics import confusion_matrix

qq=confusion_matrix(y_test,yTYPE)

qq
from sklearn.preprocessing import binarize

yBIN=binarize(y_prob,0.80)

yBIN
y_BIN1=yBIN[:,1]
yTYPE=y_BIN1.astype(int)
from sklearn.metrics import confusion_matrix

qq=confusion_matrix(y_test,yTYPE)

qq