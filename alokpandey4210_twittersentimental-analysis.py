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
df=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')

df
df.isnull().sum()
df['label'].value_counts()
X=df['tweet']

y=df['label']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
from sklearn.feature_extraction.text import CountVectorizer

vect=CountVectorizer()

X_train_dtm=vect.fit_transform(X_train)

X_test_dtm=vect.transform(X_test)

X_train_dtm.shape
from sklearn.naive_bayes import MultinomialNB

MB=MultinomialNB()

MB.fit(X_train_dtm,y_train)

MB.score(X_test_dtm,y_test)
from sklearn.linear_model import LogisticRegression

LR=LogisticRegression()

LR.fit(X_train_dtm,y_train)

LR.score(X_test_dtm,y_test)
from sklearn.neighbors import KNeighborsClassifier

KN=KNeighborsClassifier()

KN.fit(X_train_dtm,y_train)

KN.score(X_test_dtm,y_test)
from sklearn.svm import SVC

svC=SVC()

svC.fit(X_train_dtm,y_train)

svC.score(X_test_dtm,y_test)
from sklearn.tree import DecisionTreeClassifier

DT=DecisionTreeClassifier()

DT.fit(X_train_dtm,y_train)

DT.score(X_test_dtm,y_test)
from sklearn.ensemble import RandomForestClassifier

RB=RandomForestClassifier()

RB.fit(X_train_dtm,y_train)

RB.score(X_test_dtm,y_test)
from sklearn.metrics import confusion_matrix

tr=confusion_matrix(y_test,MB.predict(X_test_dtm))

tr
from sklearn.metrics import confusion_matrix

twr=confusion_matrix(y_test,RB.predict(X_test_dtm))

twr
from sklearn.metrics import confusion_matrix

tre=confusion_matrix(y_test,DT.predict(X_test_dtm))

tre
from sklearn.metrics import confusion_matrix

ec=confusion_matrix(y_test,svC.predict(X_test_dtm))

ec
from sklearn.metrics import confusion_matrix

trm=confusion_matrix(y_test,KN.predict(X_test_dtm))

trm
from sklearn.metrics import confusion_matrix

trj=confusion_matrix(y_test,LR.predict(X_test_dtm))

trj
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr,tpr,threshold=roc_curve(y_test,LR.predict_proba(X_test_dtm)[:,1])

fpr1,tpr1,threshold1=roc_curve(y_test,KN.predict_proba(X_test_dtm)[:,1])

fpr2,tpr2,threshold2=roc_curve(y_test,DT.predict_proba(X_test_dtm)[:,1])

fpr3,tpr3,threshold3=roc_curve(y_test,RB.predict_proba(X_test_dtm)[:,1])

fpr4,tpr4,threshold4=roc_curve(y_test,MB.predict_proba(X_test_dtm)[:,1])

print(fpr,tpr,threshold,fpr1,tpr1,threshold1,fpr2,tpr2,threshold2,fpr3,tpr3,threshold3,fpr4,tpr4,threshold4)
q1=roc_auc_score(y_test,LR.predict(X_test_dtm))

q2=roc_auc_score(y_test,KN.predict(X_test_dtm))

q3=roc_auc_score(y_test,DT.predict(X_test_dtm))

q4=roc_auc_score(y_test,RB.predict(X_test_dtm))

q5=roc_auc_score(y_test,MB.predict(X_test_dtm))

print(q1,q2,q3,q4,q5)

ns=DT.predict_proba(X_test_dtm)

ns
from sklearn.preprocessing import binarize

fm=binarize(ns,0.60)
sn=fm[:,1]
tx=sn.astype(int)
g1=confusion_matrix(y_test,tx)

g1
from sklearn.preprocessing import binarize

fm=binarize(ns,0.70)

sn=fm[:,1]

tx=sn.astype(int)

g2=confusion_matrix(y_test,tx)

g2
from sklearn.preprocessing import binarize

fm=binarize(ns,0.55)

sn=fm[:,1]

tx=sn.astype(int)

g3=confusion_matrix(y_test,tx)

g3
from sklearn.preprocessing import binarize

fm=binarize(ns,0.90)

sn=fm[:,1]

tx=sn.astype(int)

g4=confusion_matrix(y_test,tx)

g4
df1=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')
df1
X4=df1['tweet']

X4.shape


    
#from sklearn.feature_extraction.text import CountVectorizer

#vect=CountVectorizer()

X_test9=vect.transform(X4)

X_test9.shape
#X_tr1=X_test9.toarray()
y8=DT.predict(X_test9)
y8
df1['label']=y8
new_file=df1.to_csv('Twitter.csv')

new_file