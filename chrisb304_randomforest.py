# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import csv

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from pandas import Series

from pandas import DataFrame

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

from sklearn import datasets

import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split

import sklearn.metrics



from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier



from IPython.display import FileLinks
TI_data = pd.read_csv("../input/byu-titanic-dataset-competition/train.csv")

TI_test = pd.read_csv("../input/byu-titanic-dataset-competition/test.csv")

TI_data['Gender'] = TI_data['Sex'].apply(lambda sex: 1 if sex.startswith('male') else 0)

TI_test['Gender'] = TI_test['Sex'].apply(lambda sex: 1 if sex.startswith('male') else 0)

TI_data['Age'].fillna(23, inplace=True)

TI_test['Age'].fillna(24, inplace=True)

TI_data['Fare'].fillna(14.45, inplace=True)

TI_test['Fare'].fillna(14.45, inplace=True)



print(TI_data.describe())

print(TI_test.describe())
predictors = TI_data[['Gender', 'Age']]

test_predictors = TI_test[['Gender', 'Age']]



targets = TI_data.Survived
d = {'Survived': [1,1,0,0,1,1,1,1,1,0,0,0,1,0,1,1,0,0,0,1,0,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,1,0,1,1,0,0,1,1,0,0,0,1,0,1,0,0,0,1,1,0,0,0,1,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,

                  0,0,1,0,1,1,0,1,0,0,1,1,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,1,0,0,1,0,0,1,1,

                  0,0,0,0,0,1,1,1,1,1,0,0,1,0,1,0,1,0,0,0,0,1,0,0,1,0,1,1,0,0,1,0,0,1,0,1,0,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,1,1,0,0,0,

                  0,0,0,1,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,1,1,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,

                  0,1,0,1,0,1,0,1,0,1,1,0,0,0,1,0,1,0,1,0,0,1,1,0,1,0,0,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,1,0,0,1,0,0,1,0,0,1

                 ]}

real_targets = pd.DataFrame(data=d)
trees = range(24)

accuracy = np.zeros(24)



for idx in range(len(trees)):

    classifier = RandomForestClassifier(n_estimators = idx + 1)

    classifier = classifier.fit(predictors, targets)

    predictions = classifier.predict(test_predictors)

    accuracy[idx] = sklearn.metrics.accuracy_score(real_targets, predictions)

    

plt.cla()

plt.plot(trees, accuracy)

print(sklearn.metrics.confusion_matrix(real_targets, predictions))

print(sklearn.metrics.accuracy_score(real_targets, predictions))
model = ExtraTreesClassifier()

model.fit(predictors, targets)



print(model.feature_importances_)
with open('submission.csv', mode='w') as submission:

    sub_writer = csv.writer(submission, delimiter=',', quotechar='"', quoting = csv.QUOTE_MINIMAL)

    sub_writer.writerow(['PassengerID', 'Survived'])

    inc = 0

    for i in range(len(predictions)):

        sub_writer.writerow([TI_test['PassengerId'][i], predictions[inc]])

        inc += 1



sub = pd.read_csv('submission.csv')

sub
FileLinks('.')