# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



mydata=pd.read_csv('../input/train.csv')
#check the sample data

mydata.head()
#check for missing values

mydata.isnull().sum()
#Fill missing values for Age

mydata['Age'].fillna(mydata['Age'].median())



#Fill missing values for Cabin with Other

mydata['Cabin'].fillna('other')
#drop other missing values

mydata=mydata.dropna(axis=0)
#check for missing values

mydata.isnull().sum()
#split data to train and test

from sklearn.datasets import make_friedman1

from sklearn.feature_selection import RFE

from sklearn.svm import SVR

list(mydata.columns.values)
#convert all categorical variables using one hot encoding

#select columns which are object types



obj_df =mydata.select_dtypes(include=['object']).copy()

obj_df.columns

#Convert key categorical variables using one hot encoding

catdata=pd.get_dummies(obj_df, columns=['Sex','Embarked'])
catdata.columns.values

catdata=catdata.drop(['Ticket','Cabin'],axis=1)
mydata

#Combine the one hot encoded variables wit

catdata.columns.values

merged=mydata.merge(catdata,how='inner',on='Name')

merged.columns.values
mergeddata=merged.drop(['Sex','Embarked'],axis=1)

mergeddata.columns.values              
#split data into train and test and drop variables which are not needed for modelling

from sklearn.model_selection import train_test_split

train, test = train_test_split(mergeddata, test_size = 0.2)

train_target=train['Survived']

test_target=test['Survived']

train=train.drop(['Name','PassengerId','Survived','Ticket','Cabin'],axis=1)

test=test.drop(['Name','PassengerId','Survived','Ticket','Cabin'],axis=1)
test=test[['Age','Fare','Sex_female','Sex_male']].values

train
train.dtypes
#Feature selection using Random forest



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0,class_weight='balanced')

clf=clf.fit(train,train_target)

newlist=list(zip(clf.feature_importances_,train.columns.values))

newlist

# We identify that Sex ,Age,Pclass and Fare are the top important features predicting survival rate



from sklearn.ensemble import  GradientBoostingClassifier



clf = GradientBoostingClassifier(max_depth=2, random_state=0)

clf=clf.fit(train,train_target)

newlist=list(zip(clf.feature_importances_,train.columns.values))

data={'Importance':clf.feature_importances_,'columns':train.columns.values}

varaibleimportance=pd.DataFrame(data)
varaibleimportance
# We identify Age Gender and Fare are critical variables for predicting survival rate, this time Using Adaboost Ensemble learning

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME", n_estimators=200)

clf=clf.fit(train,train_target)

newlist=list(zip(clf.feature_importances_,train.columns.values))

data={'Importance':clf.feature_importances_,'columns':train.columns.values}

varaibleimportance=pd.DataFrame(data)
varaibleimportance


train=train[['Age','Fare','Sex_female','Sex_male']].values

test_target=test_target.values

test=test[['Age','Fare','Sex_female','Sex_male']].values
#Again Age Sex and Fare  are the important variables,using this variables and Building the final set of classification models

model=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME", n_estimators=200)

model=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME", n_estimators=200)

model=model.fit(train,train_target)

pred=model.predict(train)



pred

train_target=train_target.values

from sklearn.metrics import confusion_matrix

confusion_matrix(pred,train_target)
from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(pred, train_target)
metrics.accuracy_score(pred, train_target)
metrics.roc_auc_score(pred, train_target)
predtest=model.predict(test)
confusion_matrix(predtest,test_target)
metrics.roc_auc_score(predtest,test_target)
metrics.roc_auc_score(predtest,test_target)


#train and predict using Gradient boositng classifier

from sklearn.ensemble import  GradientBoostingClassifier

model=GradientBoostingClassifier(n_estimators=200)

model=GradientBoostingClassifier(n_estimators=200)

model=model.fit(train,train_target)

pred=model.predict(train)

from sklearn.metrics import confusion_matrix

confusion_matrix(pred,train_target)
predtest=model.predict(test)

metrics.roc_auc_score(predtest,test_target)