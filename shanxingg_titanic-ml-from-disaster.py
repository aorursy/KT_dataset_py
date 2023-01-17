# import packages

import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn import tree

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
# Import data

train = pd.read_csv("../input/train.csv", index_col='PassengerId')

test = pd.read_csv("../input/test.csv", index_col='PassengerId')

test_survived = pd.read_csv("../input/gender_submission.csv", index_col='PassengerId')
# EDA - average value of attributes based on different Pclass-Sex combination

train.groupby(['Pclass','Sex']).mean()
# EDA - average value of attributes based on different age range

age_range = pd.cut(train.Age, np.arange(0,90,10))

train.groupby(age_range).mean()
# EDA - volume of test and train data

print(f"Train Dataset:\n{train.count()}\n")

print(f"Test Dataset:\n{test.count()}")
# handle missing values of Age

train.Age.hist()

plt.show()
# handle missing values of Embarked

train.Embarked.hist()

plt.show()
# handle missing values of Fare

train.Fare.hist()

plt.show()
# Combine train x and test x

train_x = train.drop(['Survived'],axis=1)

test_x = test.copy()

attr = pd.concat([train_x, test_x])



# Preparing data for machine learning algorithm

attr = attr.drop(['Name','Cabin','Ticket'],axis=1)

attr.Age = attr.Age.fillna(attr.Age.median(skipna=True))

attr.Fare = attr.Fare.fillna(attr.Fare.median(skipna=True))

attr.Embarked = attr.Embarked.fillna(attr.Embarked.mode()[0])

le = preprocessing.LabelEncoder()

attr.Sex = le.fit_transform(attr.Sex)

attr.Embarked = le.fit_transform(attr.Embarked)



# split train x and test x

train_x = attr[:len(train_x)].copy()

test_x = attr[-len(test_x):].copy()



# extract train y and test y

train_y = train['Survived']

test_y = test_survived.copy()
# Decision Tree Model

model = tree.DecisionTreeClassifier()

model.fit(train_x, train_y)

pred_y = pd.DataFrame(model.predict(test_x), index=test_y.index, columns=['Survived'])

accuracy_score(test_y, pred_y)
# Export prediction

pred_y.to_csv('prediction.csv')