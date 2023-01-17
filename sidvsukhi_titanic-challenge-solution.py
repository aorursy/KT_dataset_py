# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

print(train_df.head())
print(train_df.info())
print(train_df.columns)
for column in train_df.columns:

    print('Total number of nan values for column {} are {}'.format(column, train_df[column].isnull().sum()))
print('Train describe: \n{}'.format(train_df.describe()))

train_mean_age = float(train_df.describe()['Age']['mean'])

train_count = float(train_df.describe()['Age']['count'])



print('Test describe: \n{}'.format(test_df.describe()))

test_mean_age = float(test_df.describe()['Age']['mean'])

test_count = float(test_df.describe()['Age']['count'])



mean_age = ((train_mean_age*train_count) + (test_mean_age*test_count))/(train_count+test_count)

print('Mean Age: {}'.format(mean_age))



train_df['Age'] = train_df['Age'].fillna(float(round(mean_age)))

test_df['Age'] = test_df['Age'].fillna(float(round(mean_age)))



for column in train_df.columns:

    print('Total number of nan values for column {} are {}'.format(column, train_df[column].isnull().sum()))
train_df.drop(['Name','PassengerId', 'Ticket'], axis=1, inplace=True)

print(train_df.head())
test_df.drop(['Name', 'Ticket'], axis=1, inplace=True)

print(test_df.head())
genders = {"male": 0, "female": 1}

data = [train_df, test_df]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)

print(train_df.head())
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train_df, test_df]



for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int)

# we can now drop the cabin feature

train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)



print(train_df)

print(test_df)
for column in train_df.columns:

    print('Total number of nan values in Train for column {} are {}'.format(column, train_df[column].isnull().sum()))

print('\n')

for column in test_df.columns:

    print('Total number of nan values in Test for column {} are {}'.format(column, test_df[column].isnull().sum()))
train_mean_fare = float(train_df.describe()['Fare']['mean'])

train_count = float(train_df.describe()['Fare']['count'])



test_mean_fare = float(test_df.describe()['Fare']['mean'])

test_count = float(test_df.describe()['Fare']['count'])



mean_fare = ((train_mean_fare*train_count) + (test_mean_fare*test_count))/(train_count+test_count)

print('Mean Age: {}'.format(mean_age))



test_df['Fare'] = test_df['Fare'].fillna(float(mean_age))



for column in test_df.columns:

    print('Total number of nan values for column {} are {}'.format(column, test_df[column].isnull().sum()))
x = train_df['Embarked'].mode()

print(x[0])

print(type(x[0]))



train_df['Embarked'] = train_df['Embarked'].fillna(x[0])



for column in train_df.columns:

    print('Total number of nan values for column in Train {} are {}'.format(column, train_df[column].isnull().sum()))
data = [train_df,test_df]

for dataset in data:

    print(dataset['Embarked'].value_counts())



EmbarkedDict = {"S":1, "C":2, "Q":3}

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(EmbarkedDict)



print(train_df.head())
data = [train_df, test_df]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)
for dataset in data:

    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print(acc_log)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(acc_random_forest)
knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, Y_train)  

Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print(acc_knn)
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, Y_train)  

Y_pred = decision_tree.predict(X_test)  

acc_dt = round(decision_tree.score(X_train, Y_train) * 100, 2)

print(acc_dt)
perceptron = Perceptron(max_iter=15)

perceptron.fit(X_train, Y_train)



Y_pred = perceptron.predict(X_test)



acc_per = round(perceptron.score(X_train, Y_train) * 100, 2)

print(acc_per)
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)



Y_pred = linear_svc.predict(X_test)



acc_svm = round(linear_svc.score(X_train, Y_train) * 100, 2)

print(acc_svm)
from sklearn.ensemble import VotingClassifier

from sklearn import model_selection



estimators = []

estimators.append(('random forest', random_forest))

estimators.append(('decision trees', decision_tree))

estimators.append(('knn', knn))

estimators.append(('svm', linear_svc))





ensemble = VotingClassifier(estimators, voting='hard')

ensemble.fit(X_train, Y_train)

Y_pred = ensemble.predict(X_test)

acc_en =  round(ensemble.score(X_train, Y_train) * 100, 2)

print(acc_en)
#Random forest is working the best!!

print(Y_prediction)

submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':Y_prediction})

print(submission.head())
filename = '/kaggle/working/Titanic_Output.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)