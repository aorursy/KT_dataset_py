import warnings

warnings.filterwarnings('ignore')

import os

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
folder = r'/kaggle/input/welcoming-party'

path_train = os.path.join(folder, 'train.csv')

path_test = os.path.join(folder, 'test.csv')

path_submission = os.path.join(folder, 'sample_submission.csv')
train = pd.read_csv(path_train)

test = pd.read_csv(path_test)

submission = pd.read_csv(path_submission)
train = train.replace('?', np.nan)

test = test.replace('?', np.nan)
num_col = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'body']

str_col = ['name', 'sex', 'cabin', 'home.dest', 'ticket', 'embarked', 'boat']
data = pd.concat([train, test], ignore_index=True)



for col in num_col:

    data[col].fillna(0, inplace=True)



for col in str_col:

    data[col].fillna('0', inplace=True)
encoder = LabelEncoder()

for column in str_col:

    data[column] = encoder.fit_transform(data[column].values)
train = data[~data['survived'].isnull()]

test = data[data['survived'].isnull()]
x_train = train.drop(columns=['idx', 'survived'])

y_train = train['survived']
model = LogisticRegression()

model.fit(x_train,y_train)

pred = model.predict(test.drop(columns=['survived','idx']))
submission['survived'] = np.array(pred).astype(int)

submission.to_csv('submission.csv', index=False)