# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing basic packages/modules

import numpy as np

import pandas as pd

from sklearn import metrics
train = pd.read_csv('/kaggle/input/titanic/train.csv', header=0)

test = pd.read_csv('/kaggle/input/titanic/test.csv', header=0)

fulldata = [train, test]
train.info()
#PClass

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()



#Sex

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()



#Family Size

for dataset in fulldata:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



#isAlone

for dataset in fulldata:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



#Embarking

for dataset in fulldata:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')



#Fare

for dataset in fulldata:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

    

#Age

for dataset in fulldata:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train['CategoricalAge'] = pd.cut(train['Age'], 5)





for dataset in fulldata:

    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)



    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare']  = 3

    dataset['Fare'] = dataset['Fare'].astype(int)    

    

    dataset.loc[ dataset['Age'] <= 16, 'Age']  = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize'], axis = 1)

test  = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize'], axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)    



X = train.iloc[:,1:]

Y = train.iloc[:,0:1]





X = X.drop(Y.index[829])

X = X.drop(Y.index[61])



Y = Y.drop(Y.index[829])

Y = Y.drop(Y.index[61])
#Splitting the dataset into Train and Test data

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.25)



#1. Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()

RFC.fit(xtrain, ytrain)

pred_rfc = RFC.predict(xtest)

acc_rfc = metrics.accuracy_score(ytest, pred_rfc)*100

#joblib.dump(RFC, 'Credit_RFC.pkl')



print('1. Using RandomForestClassifier Method')

print('Accuracy - {}'.format(acc_rfc))

print('Recall - {}'.format(metrics.recall_score(ytest, pred_rfc)))

print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_rfc)))

print('Confusion matrix')

print(metrics.confusion_matrix(ytest, pred_rfc))

 



#2. Support Vector Machines

from sklearn import svm

SVM = svm.LinearSVC(loss='hinge')

SVM.fit(xtrain, ytrain)

pred_svm = SVM.predict(xtest)

acc_svm = metrics.accuracy_score(ytest, pred_svm)*100

#joblib.dump(SVM, 'Credit_SVM.pkl')



print('2. Using SVM Method')

print('Accuracy - {}'.format(acc_svm))

print('Recall - {}'.format(metrics.recall_score(ytest, pred_svm)))

print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_svm)))

print('Confusion matrix')

print(metrics.confusion_matrix(ytest, pred_svm))

print('\n')



#3. Decision Tree

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(max_depth=10, random_state=101, max_features=None, min_samples_leaf=10)

DT.fit(xtrain, ytrain)

pred_DT = DT.predict(xtest)

acc_DT = metrics.accuracy_score(ytest, pred_DT)*100

#joblib.dump(DT, 'Credit_DT.pkl')



print('3. Using Decision Tree Method')

print('Accuracy - {}'.format(acc_DT))

print('Recall - {}'.format(metrics.recall_score(ytest, pred_DT)))

print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_DT)))

print('Confusion matrix')

print(metrics.confusion_matrix(ytest, pred_DT))

print('\n')
#For submissions

test11 = pd.read_csv('/kaggle/input/titanic/test.csv', header=0)

result = DT.predict(test)    

submission = pd.DataFrame({"PassengerId": test11["PassengerId"],"Survived": result})

submission.head(15)