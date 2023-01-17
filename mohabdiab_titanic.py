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

import matplotlib.pyplot as plt

from math import sqrt 

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import validation_curve

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, KFold, cross_val_predict, validation_curve,GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import minmax_scale

from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import  VotingClassifier

from sklearn.ensemble import GradientBoostingClassifier

import sklearn.ensemble as ens 
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/gender_submission.csv')
train.head()
train.describe()
train.info()
train.isna().sum()
train.Age.hist()
train.Age.fillna(train.Age.median(), inplace= True)
train.Fare.hist()
train.Fare.fillna(train.Fare.median(), inplace= True)
train.Embarked.value_counts()
train.Embarked.fillna('S', inplace= True)
train.isna().sum()
cabin_tr  = train.Cabin

cabin_te = test.Cabin
#cab = train[train.Cabin.notnull()]
#cab['cc'] = cab.Cabin.str.slice(start=0, stop=1)
#cab.cc.unique()
#cab['sum'] = cab.Parch + cab.SibSp

#cab['With Someone or Not'] = cab['sum'].apply(with_someone)
#cab.drop(columns=['SibSp', 'Parch', 'sum'], axis = 1, inplace= True)
#cab.Age.fillna(cab.Age.median(), inplace= True)

#cab.Fare.fillna(cab.Fare.median(), inplace= True)

#cab.Embarked.fillna('S', inplace= True)
#cab = pd.get_dummies(data = cab, columns = ['Pclass', 'Sex', 'Embarked'])

#cab.head()
#cab.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True)

#cab.head()
#le = preprocessing.LabelEncoder()

#cab['ccl'] =le.fit_transform(cab.cc)

#cab.drop(['cc'], axis=1, inplace=True)

#cab.head()
#xcab = cab[['Survived','Age','Fare','With Someone or Not','Pclass_1','Pclass_2',

 #           'Pclass_3','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']]

#ycab = cab['ccl']
#random_forest_cab = RandomForestClassifier(n_estimators=100)

#random_forest_cab.fit(xcab, ycab)

#y_pred_cab = random_forest.predict(xcab)

#accuracy_score(ycab, y_pred_cab)
train.drop('Cabin', axis=1, inplace= True)
train.head()
sns.barplot('Pclass', 'Survived', data=train, color = 'darkturquoise')

plt.show()
sns.barplot('Embarked', 'Survived', data=train, color = 'darkturquoise')

plt.show()
sns.barplot('SibSp', 'Survived', data=train, color='mediumturquoise')

plt.show()
sns.barplot('Parch', 'Survived', data=train, color='mediumturquoise')

plt.show()


sns.barplot('Sex', 'Survived', data=train, color='mediumturquoise')

plt.show()
train = pd.get_dummies(data = train, columns = ['Pclass', 'Sex', 'Embarked'])
train.head()
train.drop(columns=['Ticket', 'PassengerId'], axis=1, inplace=True)
train.head()
def with_someone(x):

    if x > 0:

        return 1

    else:

        return 0

    return x
train['sum'] = train.Parch + train.SibSp

train['With Someone or Not'] = train['sum'].apply(with_someone)
test['sum'] = test.Parch + test.SibSp

test['With Someone or Not'] = test['sum'].apply(with_someone)
train.drop(columns=['SibSp', 'Parch', 'sum'], axis = 1, inplace= True)
test.drop(columns=['SibSp', 'Parch', 'sum'], axis = 1, inplace= True)
train.isnull().sum()
test.Age.fillna(train.Age.median(), inplace= True)
test.Fare.fillna(train.Fare.median(), inplace= True)
test.Embarked.fillna('S', inplace= True)
test.drop('Cabin', axis=1, inplace= True)
passID = test.PassengerId

test.drop(columns=['Ticket', 'PassengerId'], axis=1, inplace=True)
test = pd.get_dummies(data = test, columns = ['Pclass', 'Sex', 'Embarked'])
test.head()
train.head()
#train.Age = train.Age.apply(lambda x: 1 if x >= 18 else 0)

#test.Age = test.Age.apply(lambda x: 1 if x >= 18 else 0)
def age(x):

    if x < 5:

        return 'enfant'

    elif x < 21:

        return 'teen'

    elif x < 41:

        return 'adult'

    else:

        return 'old'

    return x

    

    

    
train['Age Stage'] = train['Age'].apply(age)
train.head()
test['Age Stage'] = test['Age'].apply(age)
test.head()
train = pd.get_dummies(data=train, columns=['Age Stage'])

test = pd.get_dummies(data=test, columns=['Age Stage'])
train.head()
X = train[['Name', 'Age', 'Fare', 'With Someone or Not', 'Pclass_1', 'Pclass_2', 'Pclass_3',

       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Age Stage_adult', 'Age Stage_enfant', 

          'Age Stage_old', 'Age Stage_teen']]

Y = train['Survived']
X.head()
test.head()
train_test_data = [X, test] # combining train and test dataset



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

X['Title'].value_counts()
test['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
X.drop(columns=['Name'], axis=1, inplace=True)

test.drop(columns=['Name'], axis=1, inplace=True)
#X['Fare'] = minmax_scale(X['Fare'])

#test['Fare'] = minmax_scale(test['Fare'])

#X['Age'] = minmax_scale(X['Age'])

#test['Age'] = minmax_scale(test['Age'])



sc_X = MinMaxScaler()

X = sc_X.fit_transform(X)

test = sc_X.transform(test)

X = pd.DataFrame(X)

test = pd.DataFrame(test)
X.head()
test.head()
#X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.2)
log = LogisticRegression()

param_name = 'random_state'

param_range = [0]

train_scores, valid_scores = validation_curve(log, X, Y,

                                              param_name,param_range,cv=10)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X, Y)

y_pred = random_forest.predict(X)

accuracy_score(Y, y_pred)
param_grid = {'max_depth': np.arange(1, 10),

              'criterion': ['gini', 'entropy']}



tree = GridSearchCV(DecisionTreeClassifier(random_state =0), param_grid, scoring='accuracy')



tree.fit(X, Y)
tree.best_params_

dtree=DecisionTreeClassifier(random_state =0, criterion= 'entropy', max_depth= 6)

dtree.fit(X, Y)
tree_acc = accuracy_score(Y, dtree.predict(X))

tree_acc
X.head()
test.columns
X.head()
test.head()
svm_clf = SVC(random_state=0)

svm_parm = {'kernel': ['rbf', 'poly'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'degree': [3, 5, 7], 

            'gamma': ['auto', 'scale']}





clf = GridSearchCV(svm_clf, svm_parm, cv=5)

clf.fit(X, Y)
#pred = clf.predict(test)

#clf.best_params_
#cols = ['PassengerId', 'Survived']

#submit_df = pd.DataFrame(np.hstack((passID.values.reshape(-1,1),pred.reshape(-1,1))), 

#                        columns=cols)
#submit_df.to_csv('submission.csv', index=False)
#submit_df.head()


clf1 = RandomForestClassifier(max_depth=7, n_estimators=100)

clf1.fit(X, Y)

clf2 = GradientBoostingClassifier(n_estimators= 100, learning_rate=1.0, max_depth=1, random_state=0)

clf2.fit(X, Y)

clf3 = SVC(C=10, degree=3, gamma='auto', kernel='rbf', probability=True)

clf3.fit(X, Y)

eclf = VotingClassifier(estimators=[('RandF', clf1), ('Grad', clf2), ('SVC', clf3)], voting='soft', weights=(2.0, 2.5, 2.5))
eclf.fit(X, Y)
predf = eclf.predict(test)
cols = ['PassengerId', 'Survived']

submit_df = pd.DataFrame(np.hstack((passID.values.reshape(-1,1),predf.reshape(-1,1))), 

                         columns=cols)
#submit_df.to_csv('submission.csv', index=False)
cabin_tr = pd.DataFrame(cabin_tr)

X = pd.DataFrame(X)

cab_null = pd.concat([X, cabin_tr], axis = 1)

cab_null.head()
cab_notnull = cab_null[cab_null.Cabin.notnull()]

cab_notnull.head()
cab_notnull['cc'] = cab_notnull.Cabin.str.slice(start=0, stop=1)
le = preprocessing.LabelEncoder()#

cab_notnull['ccl'] =le.fit_transform(cab_notnull.cc)

cab_notnull.drop(['cc', 'Cabin'], axis=1, inplace=True)

cab_notnull.head()
x_cab = cab_notnull.iloc[:, 0:16]
y_cab = cab_notnull.iloc[:, -1]


clf_cab = RandomForestClassifier(max_depth=7, n_estimators=100)

clf_cab.fit(x_cab, y_cab)
cab_null = cab_null.drop(columns='Cabin')
cab_new = clf_cab.predict(cab_null)
cab_new = pd.DataFrame(cab_new)

cabfinal = pd.concat([cab_null, cab_new], axis=1)

cabfinal.head()
cab_new_test = clf_cab.predict(test)
cab_new_test = pd.DataFrame(cab_new_test)

cabfinaltest = pd.concat([test, cab_new_test], axis=1)

cabfinaltest.head()


clf1 = RandomForestClassifier(max_depth=7, n_estimators=100)

clf1.fit(cabfinal, Y)

clf2 = GradientBoostingClassifier(n_estimators= 100, learning_rate=1.0, max_depth=1, random_state=0)

clf2.fit(cabfinal, Y)

clf3 = SVC(C=10, degree=3, gamma='auto', kernel='rbf', probability=True)

clf3.fit(cabfinal, Y)

eclf = VotingClassifier(estimators=[('RandF', clf1), ('Grad', clf2), ('SVC', clf3)], voting='soft', weights=(2.0, 2.5, 2.5))
eclf.fit(cabfinal, Y)
predf = eclf.predict(cabfinaltest)
cols = ['PassengerId', 'Survived']

submit_df = pd.DataFrame(np.hstack((passID.values.reshape(-1,1),predf.reshape(-1,1))), 

                         columns=cols)
submit_df.to_csv('submission_cabin.csv', index=False)
submit_df