import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

import os

warnings.filterwarnings('ignore')

# Add the complete dataset to the repository. The data is added to ../input/ directory

!ls ../input/



#Read the first 5 headers of the dataset 

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
NanExist = False

if train.count().min() == train.shape[0] and test.count().min() == test.shape[0] :

    print('There is no missing data!') 

else:

    NanExist = True

    print('we have NAN!!!')

if NanExist == True:

    NumOfNan = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train Data', 'Test Data']) 

    print(NumOfNan[NumOfNan.sum(axis=1) > 0])
title_train = (train['Name'].str.split(',').str[1]).str.split('.').str[0]

title_test = (test['Name'].str.split(',').str[1]).str.split('.').str[0]
for i in range(0,len(title_train)): #both have same dimension

    if np.isnan(train['Age'][i]) == True:

        if 'Miss' in title_train[i] or 'Master' in title_train[i]:

            train['Age'][i] = 0

        else:train['Age'][i] = 18

for i in range(0,len(title_test)): 

    if np.isnan(test['Age'][i]) == True:

        if 'Miss' in title_test[i] or 'Master' in title_test[i]:

            test['Age'][i] = 0

        else:test['Age'][i] = 18

sum(train["Age"].isna()) # checking train

train_orig = train.copy() # save the original data 

sum(test["Age"].isna())  #checking test
train['Sex'] = train['Sex'].replace('male', 1)

train['Sex'] = train['Sex'].replace('female', 2)



test['Sex'] = test['Sex'].replace('male', 1)

test['Sex'] = test['Sex'].replace('female', 2)
fp = train['Embarked'].dropna().mode()[0]

train['Embarked'] = train['Embarked'].fillna(fp)
train['Embarked'] = train['Embarked'].map({'S': 0, 'C':1,'Q':2}).astype(int)

test['Embarked'] = test['Embarked'].map({'S': 0, 'C':1,'Q':2}).astype(int)
#Converting Pandas DataFrame to numpy arrays so that they can be used in sklearn

train_feature = train[['Sex','Age','Pclass','SibSp','Parch','Embarked']].values

train_class = train['Survived'].values

feature_names = ['Sex','Age','Pclass','SibSp','Parch','Embarked']

test_feature = test[['Sex','Age','Pclass','SibSp','Parch','Embarked']].values
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(train_feature,train_class) # This is applying the fitting



test_predict = clf.predict(test_feature) # this is the predicted RESULT

cv_score = clf.score(

    train_feature,train_class)

cv_score

from sklearn import preprocessing

poly = preprocessing.PolynomialFeatures(degree=2)

poly_train_feature = poly.fit_transform(train_feature)

poly_test_feature = poly.fit_transform(test_feature)

classfier = LogisticRegression()

classifier_ = classfier.fit(poly_train_feature, train_class)

poly_test_predict = classifier_.predict(poly_test_feature)

print(classifier_.score(poly_train_feature, train_class))

#print(classifier_.score(poly_test_feature,poly_test_predict))

LogReg_TestResult= pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':poly_test_predict})

LogReg_TestResult.head()

LogReg_TestResult.to_csv('PLogReg_TestResult.csv',index=False)
from sklearn.ensemble import RandomForestClassifier

#This has to be improved

rclf = RandomForestClassifier(criterion='gini',n_estimators=1000,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

seed= 42

rclf =RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=5, min_samples_split=2,

                           min_samples_leaf=1, max_features='auto',    bootstrap=False, oob_score=False, 

                           n_jobs=1, random_state=seed,verbose=0)

rclf.fit(train_feature,train_class)

test_predict = rclf.predict(test_feature)

print(rclf.score(train_feature, train_class))

#cv_score = rclf.score(test_feature,test_predict)

RandForst_TestResult= pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':test_predict})

RandForst_TestResult.head()

RandForst_TestResult.to_csv('RandForst_Test.csv',index=False)