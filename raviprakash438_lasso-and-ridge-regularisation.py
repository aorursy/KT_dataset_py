import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso,LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
print(os.listdir("../input"))
#Load the train dataset. It contain more then 76000 records. Lets load 10000 records only to make things fast.
df=pd.read_csv('../input/santander-customer-satisfaction/train.csv',nrows=10000)
df.shape
df.head()
df.info()
# separate dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(df.drop(labels=['TARGET'], axis=1),df['TARGET'],test_size=0.3,random_state=0)
#Filling null value with 0.
X_train.fillna(0,inplace=True)
X_test.fillna(0,inplace=True)
#Shape of training set and test set.
X_train.shape, X_test.shape
# linear models benefit from feature scaling
scaler=StandardScaler()
scaler.fit(X_train)
#Lets do the model fitting and feature selection all in single line of code.
#I will be using Logistic Regression model and select Lasso (l1 as) as a penalty
#I will be using SelectFromModel object which select the features which are non zero.
#C=1 (Inverse of regularization strength.Smaller values specify stronger regularization.)
#penalty='l1' (Specify the norm used in the penalization.Here we are using Lasso.)
sel=SelectFromModel(LogisticRegression(C=1,penalty='l1'))
sel.fit(scaler.transform(X_train),Y_train)
print('Total features-->',X_train.shape[1])
print('Selected featurs-->',sum(sel.get_support()))
print('Removed featurs-->',np.sum(sel.estimator_.coef_==0))
# create a function to build random forests and compare performance in train and test set
def RandomForest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=200, random_state=1, max_depth=4)
    rf.fit(X_train, y_train)
    print('Train set')
    pred = rf.predict_proba(X_train)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    print('Test set')
    pred = rf.predict_proba(X_test)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
#Transforming the training set and test set.
X_train_lasso=sel.transform(X_train)
X_test_lasso=sel.transform(X_test)
RandomForest(X_train_lasso,X_test_lasso,Y_train,Y_test)
#Lets do the model fitting and feature selection all in single line of code.
#I will be using Logistic Regression model and select Lasso (l1 as) as a penalty
#I will be using SelectFromModel object which select the features which are non zero.
#C=1 (Inverse of regularization strength.Smaller values specify stronger regularization.)
#penalty='l2' (Specify the norm used in the penalization.Here we are using Ridge. It is a default penalty.)
sfm=SelectFromModel(LogisticRegression(C=1,penalty='l2'))
sfm.fit(scaler.transform(X_train),Y_train)
print('Total features-->',X_train.shape[1])
print('Selected featurs-->',sum(sfm.get_support()))
print('Removed featurs-->',np.sum(sfm.estimator_.coef_==0))
np.sum(np.abs(sfm.estimator_.coef_)>np.abs(sfm.estimator_.coef_).mean())
#Transforming the training set and test set.
X_train_l2=sel.transform(X_train)
X_test_l2=sel.transform(X_test)
RandomForest(X_train_l2,X_test_l2,Y_train,Y_test)
