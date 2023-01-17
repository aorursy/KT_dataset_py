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
df=pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.head()
df.corr(method='pearson')
df['Churn'].value_counts()
X=df.drop(['customerID','Churn'],axis=1)
y=df['Churn']
X.info
from sklearn.preprocessing import LabelBinarizer

lb=LabelBinarizer()

y=lb.fit_transform(y)
X.isnull().sum()
X=pd.get_dummies(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
from sklearn.linear_model import LogisticRegression

d=LogisticRegression()

d.fit(X_train,y_train)

d.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

k=KNeighborsClassifier()

k.fit(X_train,y_train)

k.score(X_test,y_test)
from sklearn.svm import SVC

w=SVC()

w.fit(X_train,y_train)

w.score(X_test,y_test)
from sklearn.naive_bayes import GaussianNB

u=GaussianNB()

u.fit(X_train,y_train)

u.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier

m=DecisionTreeClassifier()

m.fit(X_train,y_train)

m.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

o=RandomForestClassifier()

o.fit(X_train,y_train)

o.score(X_test,y_test)
from sklearn.ensemble import ExtraTreesClassifier

i=ExtraTreesClassifier()

i.fit(X_train,y_train)

i.score(X_test,y_test)
from sklearn.linear_model import SGDClassifier

r=SGDClassifier()

r.fit(X_train,y_train)

r.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

y_predict=d.predict(X_test)

results=confusion_matrix(y_test,y_predict)

print("confusion_matrix")

print(results)

# save model

import pickle

file_names="churn.sav"

tuples=(d,X)

pickle.dump(tuples,open(file_names,'wb'))
from sklearn.metrics import confusion_matrix

y_predict1=k.predict(X_test)

results1=confusion_matrix(y_test,y_predict1)

print("confusion_matrix")

print(results1)
from sklearn.metrics import confusion_matrix

y_predict2=w.predict(X_test)

results2=confusion_matrix(y_test,y_predict2)

print("confusion_matrix")

print(results2)
from sklearn.metrics import confusion_matrix

y_predict3=m.predict(X_test)

results3=confusion_matrix(y_test,y_predict3)

print("confusion_matrix")

print(results3)
from sklearn.metrics import confusion_matrix

y_predict4=u.predict(X_test)

results4=confusion_matrix(y_test,y_predict4)

print("confusion_matrix")

print(results4)
from sklearn.metrics import confusion_matrix

y_predict5=r.predict(X_test)

results5=confusion_matrix(y_test,y_predict5)

print("confusion_matrix")

print(results5)
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr,tpr,threshold=roc_curve(y_test,k.predict_proba(X_test)[:,1])

print(fpr,'',tpr,'',threshold)

from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr1,tpr1,threshold1=roc_curve(y_test,d.predict_proba(X_test)[:,1])

print(fpr1,'',tpr1,'',threshold1)
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr2,tpr2,threshold2=roc_curve(y_test,m.predict_proba(X_test)[:,1])

print(fpr2,'',tpr2,'',threshold2)
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr3,tpr3,threshold3=roc_curve(y_test,u.predict_proba(X_test)[:,1])

print(fpr3,'',tpr3,'',threshold3)
roc_auc=roc_auc_score(y_test,k.predict(X_test))

roc_auc
roc_auc1=roc_auc_score(y_test,d.predict(X_test))

roc_auc1
roc_auc2=roc_auc_score(y_test,m.predict(X_test))

roc_auc2
roc_auc3=roc_auc_score(y_test,u.predict(X_test))

roc_auc3
yfg=d.predict_proba(X_test)
yfg
from sklearn.preprocessing import binarize

yqc=binarize(yfg,0.60)

yqc[:]
rt=yqc[:,1]
fb=rt.astype(int)
from sklearn.metrics import confusion_matrix

hj=confusion_matrix(y_test,fb)

hj
from sklearn.preprocessing import binarize

yqc=binarize(yfg,0.70)

yqc[:]

rt=yqc[:,1]

fb=rt.astype(int)

from sklearn.metrics import confusion_matrix

hj=confusion_matrix(y_test,fb)

hj
from sklearn.preprocessing import binarize

yqc=binarize(yfg,0.80)

yqc[:]

rt=yqc[:,1]

fb=rt.astype(int)

from sklearn.metrics import confusion_matrix

hj=confusion_matrix(y_test,fb)

hj
from sklearn.preprocessing import binarize

yqc=binarize(yfg,0.90)

yqc[:]

rt=yqc[:,1]

fb=rt.astype(int)

from sklearn.metrics import confusion_matrix

hj=confusion_matrix(y_test,fb)

hj