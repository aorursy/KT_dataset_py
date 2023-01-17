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
# Loading data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
# Information on data

train.info()
# Defining features and outcomes

features_train = train[['Pclass','Sex', 'SibSp','Parch']]

outcomes_train = train[['Survived']]

features_test = test[['Pclass','Sex', 'SibSp','Parch']]

features_train = pd.get_dummies(features_train)

features_test = pd.get_dummies(features_test)

features_train.head()
# Training Model

from sklearn.ensemble import AdaBoostClassifier



model = AdaBoostClassifier(random_state=40)

model.fit(features_train,outcomes_train)
# Testing Model

predictions = model.predict(features_test)
# Converting to csv

submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':predictions})

submission.to_csv('Submission.csv', index=False)