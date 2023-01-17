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
gender_submission_data = pd.read_csv("/kaggle/input/titanic/gender_submission.csv", encoding= 'unicode_escape')
test_data = pd.read_csv("/kaggle/input/titanic/test.csv", encoding= 'unicode_escape')
train_data = pd.read_csv("/kaggle/input/titanic/train.csv", encoding= 'unicode_escape')
train_y = train_data.drop('PassengerId',axis=1)
train_y = train_y.drop('Pclass',axis=1)
train_y = train_y.drop('Name',axis=1)
train_y = train_y.drop('Sex',axis=1)
train_y = train_y.drop('Age',axis=1)
train_y = train_y.drop('SibSp',axis=1)
train_y = train_y.drop('Parch',axis=1)
train_y = train_y.drop('Ticket',axis=1)
train_y = train_y.drop('Fare',axis=1)
train_y = train_y.drop('Cabin',axis=1)
train_y = train_y.drop('Embarked',axis=1)
train_data_clean = train_data.drop('Name',axis=1)
train_data_clean = train_data_clean.drop('Ticket',axis=1)
train_data_clean = train_data_clean.drop('Cabin',axis=1)
train_data_clean = train_data_clean.drop('Survived',axis=1)

test_data_clean = test_data.drop('Name',axis=1)
test_data_clean = test_data_clean.drop('Ticket',axis=1)
test_data_clean = test_data_clean.drop('Cabin',axis=1)

train_data_clean.head(1)
test_data_clean.head(1)
the_one_hot_train_x = pd.get_dummies(train_data_clean)

the_one_hot_test_x = pd.get_dummies(test_data_clean)

the_one_hot_test_x.shape
#the_one_hot_train_x = the_one_hot_train_x.drop('Embarked_C',axis=1)
the_one_hot_test_x.shape
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(np.nan_to_num(the_one_hot_train_x), train_y)
predicted_y = decision_tree.predict(np.nan_to_num(the_one_hot_test_x))
predicted_y
