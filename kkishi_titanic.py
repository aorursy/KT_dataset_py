# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas_profiling as pdp # https://qiita.com/h_kobayashi1125/items/02039e57a656abe8c48f
from sklearn import linear_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

def convert(df):
    df['IsMale'] = df['Sex'].map(lambda x: 1 if x == 'male' else 0)
    for e in df['Embarked'].dropna().unique():
        df['Embarked_' + e] = df['Embarked'].map(lambda x: 1 if x == e else 0)
    return df

df = pd.read_csv("../input/train.csv")
df = convert(df)
df.head()
pdp.ProfileReport(df)
df.describe()
def trim(df):
    df = df.drop('PassengerId', axis = 1).drop('Name', axis = 1).drop('Sex', axis = 1).drop('Ticket', axis = 1).drop('Cabin', axis = 1).drop('Embarked', axis = 1)
    df = df.drop('Age', axis = 1)
    df = df.drop('Fare', axis = 1)
    return df

# lr = linear_model.LinearRegression()
lr = linear_model.LogisticRegression()
X = trim(df).drop('Survived', axis = 1)
Y = df['Survived']

lr.fit(X, Y)
# lr.score(X, Y)
# pd.DataFrame(lr.coef_, X.columns)

test_df = pd.read_csv("../input/test.csv")
testX = trim(convert(test_df))
# test_df.describe()
p = lr.predict(testX)

submission = test_df
submission['Survived'] = [1 if x > 0.5 else 0 for x in p]
submission = submission[['PassengerId', 'Survived']]
print(submission.to_csv(index = False))
test_df = pd.read_csv("../input/test.csv")
gender_submission = pd.read_csv("../input/gender_submission.csv")

for i in range(0, len(gender_submission)):
    if submission.at[i, 'Survived'] != gender_submission.at[i, 'Survived']:
        print(i, submission.at[i, 'Survived'])
        print(test_df.take([i]))

