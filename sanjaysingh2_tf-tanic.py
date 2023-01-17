import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#load data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()

#train.info()

#train.describe()
test.head()
#Exploring data

train[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean().sort_values(by='Pclass', ascending=False)
train[['Sex', 'Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Sex', ascending=False)
train[['Parch', 'Survived']].groupby('Parch', as_index=False).mean().sort_values(by='Parch', ascending=False)
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Sex', bins=20)
train_X = train[['Pclass','Sex','Age','SibSp','Parch','Embarked', 'Fare']]

train_Y = train[['Survived']]

test_X =  test[['Pclass','Sex','Age','SibSp','Parch','Embarked','Fare']]

combine_X = [train_X, test_X]
age_frame = [train_X['Age'], test_X['Age']]

age_mean = pd.concat(age_frame, ignore_index=True).mean()



fare_frame = [train_X['Fare'], test_X['Fare']]

fare_mean = pd.concat(fare_frame, ignore_index=True).mean()



freq_port = train_X.Embarked.dropna().mode()[0]

freq_port
for dataset in combine_X:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    dataset['Age'] = dataset['Age'].fillna(age_mean)

    dataset['Fare'] = dataset['Fare'].fillna(fare_mean)

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 2, 'C': 1, 'Q': 0} ).astype(int)



train_X.head()
age_min = pd.concat(age_frame, ignore_index=True).min()

age_max = pd.concat(age_frame, ignore_index=True).max()

for dataset in combine_X:

    dataset['Age'] = (dataset['Age'] - age_mean)/(age_max-age_min)

    

train_X.head()
fare_min = pd.concat(fare_frame, ignore_index=True).min()

fare_max = pd.concat(fare_frame, ignore_index=True).max()

for dataset in combine_X:

    dataset['Fare'] = (dataset['Fare'] - fare_mean)/(fare_max-fare_min)

    

train_X.head()
train_X.info()

test_X.info()
m = int(train_X.values.shape[0] * 0.7)

dev_X = train_X.values[m:, :]

dev_Y = train_Y.values[m:, :]

train_X = train_X.values[:m, :]

train_Y = train_Y.values[:m, :]
train_X.shape, train_Y.shape, dev_X.shape, dev_Y.shape, test_X.shape
model = LogisticRegression()

model.fit(train_X, train_Y)

pred_Y = model.predict(test_X)

acc_train = round(model.score(train_X, train_Y) * 100, 2)

acc_dev = round(model.score(dev_X, dev_Y) * 100, 2)

acc_train, acc_dev
model = SVC()

model.fit(train_X, train_Y)

pred_Y = model.predict(test_X)

acc_train = round(model.score(train_X, train_Y) * 100, 2)

acc_dev = round(model.score(dev_X, dev_Y) * 100, 2)

acc_train, acc_dev
model = KNeighborsClassifier(n_neighbors = 3)

model.fit(train_X, train_Y)

pred_Y = model.predict(test_X)

acc_train = round(model.score(train_X, train_Y) * 100, 2)

acc_dev = round(model.score(dev_X, dev_Y) * 100, 2)

acc_train, acc_dev
model = GaussianNB()

model.fit(train_X, train_Y)

pred_Y = model.predict(test_X)

acc_train = round(model.score(train_X, train_Y) * 100, 2)

acc_dev = round(model.score(dev_X, dev_Y) * 100, 2)

acc_train, acc_dev
model = Perceptron()

model.fit(train_X, train_Y)

pred_Y = model.predict(test_X)

acc_train = round(model.score(train_X, train_Y) * 100, 2)

acc_dev = round(model.score(dev_X, dev_Y) * 100, 2)

acc_train, acc_dev
model = LinearSVC()

model.fit(train_X, train_Y)

pred_Y = model.predict(test_X)

acc_train = round(model.score(train_X, train_Y) * 100, 2)

acc_dev = round(model.score(dev_X, dev_Y) * 100, 2)

acc_train, acc_dev
model = SGDClassifier()

model.fit(train_X, train_Y)

pred_Y = model.predict(test_X)

acc_train = round(model.score(train_X, train_Y) * 100, 2)

acc_dev = round(model.score(dev_X, dev_Y) * 100, 2)

acc_train, acc_dev
model = DecisionTreeClassifier()

model.fit(train_X, train_Y)

pred_Y = model.predict(test_X)

acc_train = round(model.score(train_X, train_Y) * 100, 2)

acc_dev = round(model.score(dev_X, dev_Y) * 100, 2)

acc_train, acc_dev
model = RandomForestClassifier(n_estimators=100)

model.fit(train_X, train_Y)

pred_Y = model.predict(test_X)

acc_train = round(model.score(train_X, train_Y) * 100, 2)

acc_dev = round(model.score(dev_X, dev_Y) * 100, 2)

acc_train, acc_dev
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": pred_Y

    })
submission.to_csv('submission.csv', index=False)