# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df
def preprocess(df):
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna('Unknown')
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, 'Unknown': 3} ).astype(int)
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    df['Cabin'] = pd.get_dummies(df['Cabin'])
    df['Ticket'] = df['Ticket'].fillna('Unknown')
    df['Ticket'] = pd.get_dummies(df['Ticket'])
    df = df.drop(['Name','PassengerId'],axis=1)
    return df
df1 = preprocess(df)
df1 = df1.sample(frac=1).reset_index(drop=True)
from sklearn.model_selection import train_test_split
x = df1.drop('Survived', axis=1)
y = df1.Survived
train_x, test_x, train_y, test_y = train_test_split(x, y,train_size=0.7)
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
clf = clf.fit(train_x, train_y)
pred = clf.predict(test_x)
print(accuracy_score(pred, test_y))
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df1_test = preprocess(df_test)
submit_data =  pd.Series(clf.predict(df1_test), name='Survived', index=df_test['PassengerId'])
submit_data.to_csv('submit.csv', header=True)
