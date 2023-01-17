import os

print(os.listdir("../input"))



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
features = pd.read_csv('../input/features.txt',header=None,sep=' ',names=('ID','Activity'))

labels = pd.read_csv('../input/activity_labels.txt',header=None,sep=' ',names=('ID','Sensor'))



X_train = pd.read_table('../input/X_train.txt',header=None,sep='\s+')

y_train = pd.read_table('../input/y_train.txt',header=None,sep='\s+')



X_test = pd.read_table('../input/X_test.txt',header=None,sep='\s+')

y_test = pd.read_table('../input/y_test.txt',header=None,sep='\s+')



train_sub = pd.read_table('../input/subject_train.txt',header=None,names=['SubjectID'])

test_sub = pd.read_table('../input/subject_test.txt',header=None,names=['SubjectID'])
X_train.head()
X_train.columns = features.iloc[:,1]

X_test.columns = features.iloc[:,1]
y_train.columns = ['Activity']

y_test.columns = ['Activity']
X_train['SubjectID'] = train_sub

X_test['SubjectID'] = test_sub
X_train.head()
y_train.head()
X_train.isnull().sum().max()
X_test.isnull().sum().max()
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error
from sklearn.model_selection import KFold,StratifiedKFold 
from sklearn.model_selection import RepeatedKFold
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)
print(rmsle(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
random_forest = RandomForestClassifier(n_estimators=6)
#kfold = KFold().split(X_train,y_train)
#for k, train in enumerate(kfold):

#    random_forest.fit(X_train[train], y_train[train])
X_train.head()
#Rp_Kfold = RepeatedKFold(n_splits=5).split(X_train,y_train)



#for k, train in enumerate(Rp_Kfold):

#    random_forest.fit(X_train[train],y_train[train])
#St_Kfold = StratifiedKFold(n_splits=5).split(X_train,y_train)



#for k, train in enumerate(St_Kfold):

#    random_forest.fit(X_train[train],y_train[train])
random_forest.fit(X_train,y_train)
y_pred = random_forest.predict(X_test)
print(rmsle(y_test,y_pred))

print(confusion_matrix(y_pred,y_test))

print(classification_report(y_pred,y_test))
print('Using KFold cross validation technique')

print(rmsle(y_test,y_pred))

print(confusion_matrix(y_pred,y_test))

print(classification_report(y_pred,y_test))
gra_boost = GradientBoostingClassifier()
gra_boost.fit(X_train,y_train)

pred = gra_boost.predict(X_test)
print(rmsle(y_test,pred))

print(confusion_matrix(pred,y_test))

print(classification_report(pred,y_test))