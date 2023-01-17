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
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn import datasets

from sklearn.metrics import accuracy_score



iris=datasets.load_iris()



X=iris.data[:,:2]

Y=iris.target



X.shape



X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.25,random_state=7)



print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)



X_train_2,X_test_2,Y_train_2,Y_test_2= train_test_split(X_train,Y_train,test_size=0.25,random_state=7)



print(X_train_2.shape,X_test_2.shape,Y_train_2.shape,Y_test_2.shape)



svc_clf=SVC(kernel='linear',random_state=7)

svc_clf.fit(X_train_2,Y_train_2)



lr_clf=LogisticRegression()

lr_clf.fit(X_train_2,Y_train_2)



svc_pred=svc_clf.predict(X_test_2)

lr_pred=lr_clf.predict(X_test_2)



print(svc_pred)

print(lr_pred)



acc_svc=accuracy_score(Y_test_2,svc_pred)



print("Accuracy of SVC: ",acc_svc)



acc_lr=accuracy_score(Y_test_2,lr_pred)



print("Accuracy of Logistic Regression: ", acc_lr)



print("the accuracy of svc on original test set: ",accuracy_score(svc_clf.predict(X_test),Y_test))



#Cross_validation



from sklearn.model_selection import cross_val_score





svc_scores=cross_val_score(svc_clf,X_train,Y_train,cv=4)



svc_scores



print("Average cross-validation score for SVC: ",svc_scores.mean())

print("Standard Deviation: ",svc_scores.std())



lr_scores=cross_val_score(lr_clf,X_train,Y_train,cv=4)



lr_scores



print("Average cross-validation score for logistic regression: ",lr_scores.mean())

print("Standard Deviation:",lr_scores.std())



from sklearn.model_selection import StratifiedKFold



skf=StratifiedKFold(n_splits=4)



svc_score_stratified=cross_val_score(svc_clf,X_train,Y_train,cv=skf)



print("Average score of SVC using StratifiedKFold:",svc_score_stratified.mean())

print("Standard Deviation:",svc_score_stratified.std())
