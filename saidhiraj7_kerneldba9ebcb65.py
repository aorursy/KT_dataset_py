# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
X=np.loadtxt('../input/secom_data.txt')
f=open('../input/secom_labels.txt')
y=np.array([int(line.split()[0]) for line in f.readlines()[:550]])
y[y==-1]=0
X=pd.DataFrame(X)
X=X.fillna(X.mean(),axis=0)
X=X.dropna(axis=1)
X.shape
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
cv_score_mean=[]
acc_score=[]
y_pred=[]
roc_auc_scores=[]
models_list=[LogisticRegression(solver='sag'),SVC(gamma='auto',probability=True),RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0),DecisionTreeClassifier(random_state=0),KNeighborsClassifier(n_neighbors=5)]
for model in models_list:
    pipeline=Pipeline([('scaler',StandardScaler()),('clf',model)])
    cvx = KFold(n_splits=10, shuffle=False, random_state=0)
    mean_cv_score = cross_val_score( pipeline, X_train, y_train, cv=cvx).mean()
    cv_score_mean.append(mean_cv_score)
    pipeline.fit(X_train,y_train)
    print('Training set accuracy_score: ',accuracy_score(pipeline.predict(X_train),y_train))
    pred=pipeline.predict(X_test)
    acc=accuracy_score(pred,y_test)
    print('Cross Validation score :',mean_cv_score)
    print('Test set accuracy_score:',acc)
    pred_proba=pipeline.predict_proba(X_test)[:,1]
    rocaucscore=roc_auc_score(y_test,pred_proba)
    print('roc_auc_score',rocaucscore)
    roc_auc_scores.append(rocaucscore)
   
    acc_score.append(acc)
    y_pred.append(pred)
from sklearn.metrics import confusion_matrix
conf_mats=[]
for pred in y_pred:
    conf_mat = confusion_matrix(y_test,pred)
    print(conf_mat)
    conf_mats.append(conf_mat)
# KNN is the best Classifier followed by RandomForestClassifier