import numpy as np

import pandas as pd

import random

import csv

from sklearn import svm
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

features = list(test)

train.head()
unwanted_features = ['Name', 'Ticket', 'Cabin']

train = train.drop(unwanted_features, axis=1)

test = test.drop(unwanted_features, axis=1)

train_test = [train, test]

features = list(set(features) - set(unwanted_features))
for feature in features:

    if test[feature].hasnans or train[feature].hasnans:

        print (feature)
for dataset in train_test:

    age_mean = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    random_age = np.random.randint(age_mean - age_std, age_mean + age_std, age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = random_age

    

    fare_mean = dataset['Fare'].mean()

    fare_std = dataset['Fare'].std()

    fare_null_count = dataset['Fare'].isnull().sum()

    random_fare = np.random.randint(fare_mean - fare_std, fare_mean + fare_std, fare_null_count)

    dataset['Fare'][np.isnan(dataset['Fare'])] = random_fare

    

    embarked_keys = pd.get_dummies(dataset['Embarked'])

    dataset['Embarked'] = dataset['Embarked'].fillna(random.sample(list(embarked_keys), 1)[0])
for dataset in train_test:

    dataset['Sex'] = pd.Categorical(dataset['Sex']).codes

    dataset['Embarked'] = pd.Categorical(dataset['Embarked']).codes
train.head()
for feature in features:

    if test[feature].hasnans or train[feature].hasnans:

        print (feature)
classes = 4

for dataset in train_test:

    age_min = np.min(dataset['Age'])

    age_max = np.max(dataset['Age'])

    age_classes =  np.linspace(age_min, age_max, classes)

    dataset['Age'] = np.digitize( dataset['Age'], age_classes)

    

    fare_min = np.min(dataset['Fare'])

    fare_max = np.max(dataset['Fare'])

    fare_classes =  np.linspace(fare_min, fare_max, classes)

    dataset['Fare'] = np.digitize(dataset['Fare'], fare_classes)
train.head()
for dataset in train_test:

    for each_class in range(classes):

        print ("class " , each_class , " : " , list(dataset['Fare']).count(each_class))
train_x = train.drop(['PassengerId','Survived'], axis=1)

train_y = train['Survived']



test_x = test.drop('PassengerId', axis=1)
classifier = svm.SVC()

classifier.fit(train_x, train_y)
predicted = classifier.predict(test_x)
passengerid = ['PassengerId'] + list(test['PassengerId'])

survived = ['Survived'] + list(predicted)
with open('predicted.csv', 'w', newline='') as fp:

    writer = csv.writer(fp, delimiter=',')

    writer.writerows(zip(passengerid , survived))