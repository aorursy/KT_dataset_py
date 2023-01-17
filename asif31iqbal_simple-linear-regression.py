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
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_x = train_df.drop(['PassengerId', 'Survived'], axis=1)
test_x = test_df.drop(['PassengerId'], axis=1)
train_x['Sex'] = train_x.Sex.apply(lambda x: 0 if x == 'male' else 1)
test_x['Sex'] = test_x.Sex.apply(lambda x: 0 if x == 'male' else 1)
train_x = train_x[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
       'Cabin', 'Embarked']]
test_x = test_x[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
       'Cabin', 'Embarked']]
train_x['Cabin'] = train_x.Cabin.fillna('').apply(lambda x: x[0] if x else '')
test_x['Cabin'] = test_x.Cabin.fillna('').apply(lambda x: x[0] if x else '')
train_y = train_df['Survived']
train_x_encoded = pd.get_dummies(train_x)
test_x_encoded = pd.get_dummies(test_x)
train_x_encoded, test_x_encoded = train_x_encoded.align(test_x_encoded, join='left', axis=1)
imputer = Imputer()
train_x_imputed = imputer.fit_transform(train_x_encoded)
test_x_encoded = test_x_encoded.fillna(0)
test_x_imputed = imputer.fit_transform(test_x_encoded)
log_reg_model = LogisticRegression()
log_reg_model.fit(train_x_imputed, train_y)
log_reg_result = log_reg_model.predict(test_x_imputed)
my_submission = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': log_reg_result})
my_submission.to_csv('submission.csv', index=False)
