from pandas import Series, DataFrame

import pandas as pd

import numpy as np

import os

import matplotlib.pylab as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

import sklearn.metrics

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
test_data = pd.read_csv("../input/titanic/test.csv")

training_data = pd.read_csv("../input/titanic/train.csv")

result = pd.read_csv("../input/titanic/gender_submission.csv")



training_data = training_data.fillna(-1)

test_data = test_data.fillna(-1)



train_data, validation_data = train_test_split(training_data, test_size=.1)
train_data.head()
explanatory_variables = ['Pclass', 'Sex', 'Age', 'SibSp', 'Embarked']

response_variables = 'Survived'
features_train = train_data[explanatory_variables]

features_train = features_train.replace('male', 0)

features_train = features_train.replace('female', 1)

features_train = features_train.replace('S', 0)

features_train = features_train.replace('C', 1)

features_train = features_train.replace('Q', 1)

target_train = train_data[response_variables]



features_eval = validation_data[explanatory_variables]

features_eval = features_eval.replace('male', 0)

features_eval = features_eval.replace('female', 1)

features_eval = features_eval.replace('S', 0)

features_eval = features_eval.replace('C', 0)

features_eval = features_eval.replace('Q', 0)

target_eval = validation_data[response_variables]



features_training = training_data[explanatory_variables]

features_training = features_training.replace('male', 0)

features_training = features_training.replace('female', 1)

features_training = features_training.replace('S', 0)

features_training = features_training.replace('C', 0)

features_training = features_training.replace('Q', 0)

target_training = training_data[response_variables]



features_test = test_data[explanatory_variables]

features_test = features_test.replace('male', 0)

features_test = features_test.replace('female', 1)

features_test = features_test.replace('S', 0)

features_test = features_test.replace('C', 0)

features_test = features_test.replace('Q', 0)
accuracy = np.zeros(300)

n_estimators_array = range(300)
for i in n_estimators_array:

    classifier = RandomForestClassifier(n_estimators = i+1)

    decision_tree = classifier.fit(features_training, target_training)

    predictions = decision_tree.predict(features_test)

    accuracy[i] = sklearn.metrics.accuracy_score(result['Survived'], predictions)
plt.cla()

plt.plot(n_estimators_array, accuracy)
classifier = RandomForestClassifier(n_estimators = 125)

decision_tree = classifier.fit(features_training, target_training)

predictions = decision_tree.predict(features_test)

sklearn.metrics.accuracy_score(result['Survived'], predictions)
submit = pd.DataFrame(list(zip(test_data['PassengerId'], predictions)), columns = ['PassengerId', 'Survived'])

submit.to_csv('submission.csv', index=False)