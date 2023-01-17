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
df=pd.read_csv('/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')

df
df['Category'].value_counts()
df.info()
df['Category'].value_counts()
X=df['Message']
y=df['Category']
from sklearn.preprocessing import LabelBinarizer

lb=LabelBinarizer()

y=lb.fit_transform(y)

y[0:5]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y)
from sklearn.feature_extraction.text import CountVectorizer

vect=CountVectorizer()

X_train_dtm=vect.fit_transform(X_train)

X_test_dtm=vect.transform(X_test)
from sklearn.svm import SVC

MB=SVC()

MB.fit(X_train_dtm,y_train)

MB.score(X_test_dtm,y_test)
from sklearn.naive_bayes import MultinomialNB

ds=MultinomialNB()

ds.fit(X_train_dtm,y_train)

ds.score(X_test_dtm,y_test)
from sklearn.neighbors import KNeighborsClassifier

KN=KNeighborsClassifier()

KN.fit(X_train_dtm,y_train)

KN.score(X_test_dtm,y_test)
from sklearn.linear_model import LogisticRegression

LM=LogisticRegression()

LM.fit(X_train_dtm,y_train)

LM.score(X_test_dtm,y_test)
from sklearn.svm import SVC

vs=SVC()

vs.fit(X_train_dtm,y_train)

vs.score(X_test_dtm,y_test)
from sklearn.tree import DecisionTreeClassifier

td=DecisionTreeClassifier()

td.fit(X_train_dtm,y_train)

td.score(X_test_dtm,y_test)
from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier()

RF.fit(X_train_dtm,y_train)

RF.score(X_test_dtm,y_test)
y1=MB.predict(X_test_dtm[0:2])

y2=ds.predict(X_test_dtm[0:2])

y3=KN.predict(X_test_dtm[0:2])

y4=LM.predict(X_test_dtm[0:2])

y5=vs.predict(X_test_dtm[0:2])

y7=RF.predict(X_test_dtm[0:2])

print(y1,y2,y3,y4,y5,y7)
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr,tpr,threshold=roc_curve(y_test,ds.predict_proba(X_test_dtm)[:,1])

fpr1,tpr1,threshold1=roc_curve(y_test,KN.predict_proba(X_test_dtm)[:,1])

fpr2,tpr2,threshold2=roc_curve(y_test,LM.predict_proba(X_test_dtm)[:,1])

fpr3,tpr3,threshold3=roc_curve(y_test,RF.predict_proba(X_test_dtm)[:,1])

print(fpr,tpr,threshold,fpr1,tpr1,threshold1,fpr2,tpr2,threshold2,fpr3,tpr3,threshold3)
gh=roc_auc_score(y_test,LM.predict(X_test_dtm))

av=roc_auc_score(y_test,MB.predict(X_test_dtm))

qv=roc_auc_score(y_test,ds.predict(X_test_dtm))

cb=roc_auc_score(y_test,KN.predict(X_test_dtm))

wc=roc_auc_score(y_test,RF.predict(X_test_dtm))

print(gh,av,qv,cb,wc)
bw=ds.predict_proba(X_test_dtm)

bw
rsv=bw.astype(int)
gz=rsv[:,1]

gz
import pickle 

tuples=(ds,X)

file_name='spam.ipynb'

pickle.dump(tuples,open(file_name,'wb'))