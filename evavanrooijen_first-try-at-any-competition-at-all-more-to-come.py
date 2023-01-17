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
test_path='../input/test.csv'

train_path='../input/train.csv'

test=pd.read_csv(test_path)

train=pd.read_csv(train_path)



print(train.columns)

print(test.columns)
train.describe()

train.nunique()


features=['Pclass', 'Sex', 'Embarked']



#features=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

train_y=train.Survived

train_X=train[features]

val_X=test[features]
train_X.dtypes
one_hot_encoded_training_predictors = pd.get_dummies(train_X)
one_hot_encoded_training_predictors.dtypes
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()

model.fit(one_hot_encoded_training_predictors, train_y)
predictions=model.predict(pd.get_dummies(test[features]))

print(predictions.size)

my_first_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})



my_first_submission.to_csv('submission.csv', index=False)