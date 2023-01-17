# This Python 3 environment comes with many helpful analytics libraries installed



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import time

import math

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/titanic/train.csv")

train.head()
test = pd.read_csv("../input/titanic/test.csv")

test.head()
train.info()

train.PassengerId.isnull().values.any()
test.info()

test.PassengerId.isnull().values.any()
all = pd.concat([train, test], sort = False)

all.info()
#fill missing values with average

all['Age'] = all['Age'].fillna(value=all['Age'].median())

all['Fare'] = all['Fare'].fillna(value=all['Fare'].median())

all.info()
sns.catplot(x = 'Embarked', kind = 'count', data = all) #or all['Embarked'].value_counts()
all['Embarked'] = all['Embarked'].fillna('S')

all.info()
#dividing men, women, children

all.loc[:,'who'] = 'M'

#1 for child, 2 for woman, 3 for man

all.loc[ all['Age'] <= 16, 'who'] = 'C'

all.loc[ (all['Age'] > 16) & (all['Sex'] == 'female'), 'who'] = 'W'

all.head()
g = sns.catplot(x='who', y='Survived', col='Pclass',

                data=all, saturation=.5,

                kind='bar', ci=None, aspect=.6)

(g.set_axis_labels('', 'Survival Rate')

  .set_xticklabels(['Men', 'Women', 'Children'])

  .set_titles('1st Class', '2nd Class', '3rd Class')

  .set(ylim=(0, 1))

  .despine(left=True))  
#Title

import re

def get_title(name):

    title_search = re.search(' ([A-Za-z]+\.)', name)

    

    if title_search:

        return title_search.group(1)

    return ""
all['Title'] = all['Name'].apply(get_title)

all['Title'].value_counts()
all['Title'] = all['Title'].replace(['Capt.', 'Dr.', 'Major.', 'Rev.'], 'Officer.')

all['Title'] = all['Title'].replace(['Lady.', 'Countess.', 'Don.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Royal.')

all['Title'] = all['Title'].replace(['Mlle.', 'Ms.'], 'Miss.')

all['Title'] = all['Title'].replace(['Mme.'], 'Mrs.')

all['Title'].value_counts()

#Cabin

all['Cabin'] = all['Cabin'].fillna('Missing')

all['Cabin'] = all['Cabin'].str[0]

all['Cabin'].value_counts()
#Family Size Column

all['Family Size'] = all['SibSp'] + all['Parch'] + 1

all['IsAlone'] = 0

all.loc[all['Family Size']==1, 'IsAlone'] = 1

all.head()
#Drop Name and ticket columns

all = all.drop(['Name', 'Ticket'], axis = 1)

all.head()
#create dummy variables

all = pd.get_dummies(all, drop_first = True)

for col in all.columns: 

    print(col) 
all.head()
#train data from all

all_train = all[all['Survived'].notna()]

all_train.info()
#test data from all

all_test = all[all['Survived'].isna()]

all_test.info()
#train and test split

from sklearn.model_selection import train_test_split

X = all_train.drop(['PassengerId', 'Survived'], axis = 1)

y = all_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
#applying standard scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 200)

rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)
#check model performance

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, pred_rfc))

print(confusion_matrix(y_test, pred_rfc))
from sklearn.metrics import accuracy_score

cm = accuracy_score(y_test, pred_rfc)

print(cm)
#determining best n estimators

best_n = 0

best_cm = 0

#for n_est in range(100, 500):

#    rfc = RandomForestClassifier(n_estimators = n_est)

#    rfc.fit(X_train, y_train)

#    pred_rfc = rfc.predict(X_test)

#    cm = accuracy_score(y_test, pred_rfc)

#    if(cm > best_cm):

#        best_cm = cm

#        best_n = n_est

best_n = 255

accuracy = 0.789

print('best n = ', best_n)

print('accuracy = ', best_cm)
from sklearn.svm import SVC

from sklearn import svm

clf = svm.SVC()

clf.fit(X_train, y_train)

pred_clf = clf.predict(X_test)
print(classification_report(y_test, pred_clf))

print(confusion_matrix(y_test, pred_clf))
print(accuracy_score(y_test, pred_clf))
from sklearn.neural_network import MLPClassifier

mlpc = MLPClassifier(hidden_layer_sizes = (2, 4), max_iter = 700)

mlpc.fit(X_train, y_train)

pred_mlpc = mlpc.predict(X_test)
#check model performance

print(classification_report(y_test, pred_mlpc))

print(confusion_matrix(y_test, pred_mlpc))
print(accuracy_score(y_test, pred_mlpc))
#determining best layer sizes

best_i = 0

best_j = 0

best_acc = 0

#for i in range(1, 20):

#    for j in range(1, 20):

#        mlpc = MLPClassifier(hidden_layer_sizes = (i,j), max_iter = 700)

#        mlpc.fit(X_train, y_train)

#        pred_mlpc = mlpc.predict(X_test)

#        acc = accuracy_score(y_test, pred_mlpc)

#        if(acc > best_acc):

#            best_acc = acc

#            best_i = i

#            best_j = j

best_i = 2

best_j = 4

best_acc = 0.8206

print(best_i, best_j, best_acc)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver = 'liblinear')

logmodel.fit(X_train,y_train)

pred_logmodel = logmodel.predict(X_test)
print(classification_report(y_test,pred_logmodel))

print(confusion_matrix(y_test,pred_logmodel))
print(accuracy_score(y_test, pred_logmodel))
TestForPred = all_test.drop(['PassengerId', 'Survived'], axis = 1)

TestForPred.info()
t_pred = logmodel.predict(TestForPred).astype(int)

PassengerId = all_test['PassengerId']

logSub = pd.DataFrame({'PassengerId': PassengerId, 'Survived':t_pred })

logSub.head()
logSub.to_csv('/kaggle/workingLogistic_Regression_Submission.csv', index = False)