# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab as pl

import scipy.optimize as opt

from sklearn import preprocessing

%matplotlib inline 

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#load the test data

test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.isnull().sum()
#rather than fill the blank values for age with the mean of the column, it's better to find the mean for each title as that's likely to be closer to the actual value



test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(test['Title'], test['Sex'])

test = test.drop('Name',axis=1)
test['Title'].unique()
#tidy up some of the more obscure values for title



test['Title'] = np.where((test.Title=='Capt') | (test.Title=='Countess') | (test.Title=='Don') | (test.Title=='Dona')

                        | (test.Title=='Jonkheer') | (test.Title=='Lady') | (test.Title=='Sir') | (test.Title=='Major') | (test.Title=='Rev') | (test.Title=='Col'),'Other',test.Title)



test['Title'] = test['Title'].replace('Ms','Miss')

test['Title'] = test['Title'].replace('Mlle','Miss')

test['Title'] = test['Title'].replace('Mme','Mrs')
#display the mean Ages for 

test[['Title','Age']].groupby(['Title']).mean()
#replace empty values for age with the mean for the title

test['Age'] = np.where((test.Age.isnull()) & (test.Title=='Master'),7,

                        np.where((test.Age.isnull()) & (test.Title=='Miss'),22,

                                 np.where((test.Age.isnull()) & (test.Title=='Mr'),32,

                                          np.where((test.Age.isnull()) & (test.Title=='Mrs'),39,

                                                  np.where((test.Age.isnull()) & (test.Title=='Other'),42,

                                                           np.where((test.Age.isnull()) & (test.Title=='Dr'),53,test.Age)))))) 
#replace SibSp with travel alone dummy



test['TravelAlone']=np.where((test["SibSp"]+test["Parch"])>0, 0, 1)

test["Fare"].fillna(test["Fare"].median(skipna=True), inplace=True)

test=pd.get_dummies(test, columns=["Pclass","Embarked","Sex"])
#drop non-numeric/unnecessary columns



test.drop('Sex_female', axis=1, inplace=True)

test.drop('Ticket', axis=1, inplace=True)

test.drop('Title', axis=1, inplace=True)

test.drop('Cabin', axis=1, inplace=True)

test.drop('SibSp', axis=1, inplace=True)

test.drop('Parch', axis=1, inplace=True)
test.head()
#load the training dataset and do the same operations to engineer the columns



train = pd.read_csv("/kaggle/input/titanic/train.csv")
train['Cabin'] = np.where((train.Pclass==1) & (train.Cabin=='U'),'C',

                                            np.where((train.Pclass==2) & (train.Cabin=='U'),'D',

                                                                        np.where((train.Pclass==3) & (train.Cabin=='U'),'G',

                                                                                                    np.where(train.Cabin=='T','C',train.Cabin))))
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])

train = train.drop('Name',axis=1)
train['Title'] = np.where((train.Title=='Capt') | (train.Title=='Countess') | (train.Title=='Don') | (train.Title=='Dona')

                        | (train.Title=='Jonkheer') | (train.Title=='Lady') | (train.Title=='Sir') | (train.Title=='Major') | (train.Title=='Rev') | (train.Title=='Col'),'Other',train.Title)



train['Title'] = train['Title'].replace('Ms','Miss')

train['Title'] = train['Title'].replace('Mlle','Miss')

train['Title'] = train['Title'].replace('Mme','Mrs')
train['TravelAlone']=np.where((train["SibSp"]+train["Parch"])>0, 0, 1)

train["Fare"].fillna(train["Fare"].median(skipna=True), inplace=True)



train=pd.get_dummies(train, columns=["Pclass","Embarked","Sex"])
train.drop('Sex_female', axis=1, inplace=True)

train.drop('PassengerId', axis=1, inplace=True)

train.drop('Ticket', axis=1, inplace=True)

train.drop('Cabin', axis=1, inplace=True)

train.drop('SibSp', axis=1, inplace=True)

train.drop('Parch', axis=1, inplace=True)
train['Age'] = np.where((train.Age.isnull()) & (train.Title=='Master'),7,

                        np.where((train.Age.isnull()) & (train.Title=='Miss'),22,

                                 np.where((train.Age.isnull()) & (train.Title=='Mr'),32,

                                          np.where((train.Age.isnull()) & (train.Title=='Mrs'),39,

                                                  np.where((train.Age.isnull()) & (train.Title=='Other'),42,

                                                           np.where((train.Age.isnull()) & (train.Title=='Dr'),53,train.Age))))))    
train.drop('Title', axis=1, inplace=True)
train.head()
#import necessary packages and libraries



from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss



#determine columns used for logistic regression

cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male"]



# create X (features) and y (response)

X = train[cols]

y = train['Survived']



# use train/test split with different random_state values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)



# check classification scores of logistic regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

y_pred_proba = logreg.predict_proba(X_test)[:, 1]

[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)

print('Train/Test split results:')

print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))

print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))

print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
test['Survived'] = logreg.predict(test[cols])

test.head()
#create submission dataframe

submission = test[['PassengerId', 'Survived']]

submission.head()
#save to csv

submission.to_csv('Submission.csv', index=False)