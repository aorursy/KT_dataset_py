# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
scaler=StandardScaler()
sourceDataFile=os.listdir("../input")
f=open("../input/"+sourceDataFile[0])
creditDF=pd.read_csv(f)
X=creditDF.iloc[:,0:30]

y=creditDF.iloc[:,30:31]
scaledX=pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
pd.plotting.scatter_matrix(scaledX.corr(),figsize=(30,30))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.80,random_state=43)
X_train.shape,y_train.shape
from sklearn.linear_model import LogisticRegression
logRegModel=LogisticRegression()

logRegModel.fit(X_train,y_train)
from sklearn.svm import LinearSVC
svc=LinearSVC(random_state=43)
svc.fit(X_train,y_train)
print('Logistic regression accuracy:',logRegModel.score(X_test,y_test)*100)

print('SVC accuracy:',svc.score(X_test,y_test)*100)
from sklearn.ensemble import GradientBoostingClassifier
gbClf=GradientBoostingClassifier()

gbClf.fit(X_train,y_train)
print('GradientBoost accuracy:',gbClf.score(X_test,y_test)*100)
from sklearn.metrics import roc_auc_score

print('GradientBoost roc_auc score:',roc_auc_score(y_true=y_test,y_score=gbClf.predict(X_test))*100)

print('Logistic Regression roc_auc score:',roc_auc_score(y_true=y_test,y_score=logRegModel.predict(X_test))*100)

print('SVM roc_auc score:',roc_auc_score(y_true=y_test,y_score=svc.predict(X_test))*100)
from xgboost import XGBClassifier,plot_importance
xgbClf=XGBClassifier(n_estimators=1000)
xgbClf.fit(X_train,y_train)
plot_importance(xgbClf)
xgbClf.score(X_test,y_test)*100