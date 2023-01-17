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
df = pd.read_csv('../input/train.csv')

y = df['Survived']

print(df.head())

print(df.dtypes)

print(df.isnull().any())
from sklearn import model_selection

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

import numpy as np
age_label =  LabelEncoder()

fare_label = LabelEncoder()

sex_label = LabelEncoder()

embark_label = LabelEncoder()

title_label = LabelEncoder()

cabin_label = LabelEncoder()
def feature_engineering(df):

    # 数値化

    c = df.copy()

    c['Age'].fillna(c['Age'].dropna().median(), inplace=True)

    c['AgeBucket'] = pd.cut(c['Age'], bins=5, labels=False)

    c['AgeCode'] = age_label.fit_transform(c['AgeBucket'])

    c = c.drop(columns=['Age', 'AgeBucket'])

    c['Fare'].fillna(c['Fare'].dropna().median())

    c['FareBucket'] = pd.cut(c['Fare'], bins=4, labels=False)

    c['FareCode'] = fare_label.fit_transform(c['FareBucket'])

    c = c.drop(columns=['Fare','FareBucket'])

    c['SexCode'] = sex_label.fit_transform(c['Sex'])

    c = c.drop(columns=['Sex'])

    # nanがあるのでなにかで埋める

    c['Embarked'].fillna(c['Embarked'].mode()[0], inplace=True)

    # Embarkedを数値化する

    c['EmbarkedCode'] = embark_label.fit_transform(c['Embarked'])

    c = c.drop(columns=['Embarked'])

    # 数値化されていることを確認

    c['Title'] = c['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

    title_names = (c['Title'].value_counts() < 10)

    c['Title'] = c['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

    c['TitleCode'] = title_label.fit_transform(c['Title'])

    c = c.drop(columns=['Name', 'Title'])

    c['CabinClass'] = c['Cabin'].apply(lambda x: x[0] if type(x) == str and len(x) > 0 else 'UNKOWN')

    #c['Cabin'].fillna(c['Cabin'].mode()[0], inplace=True)

    c['Cabin'].fillna('UNKOWN', inplace=True)

    c['CabinCode'] = cabin_label.fit_transform(c['CabinClass'])

    c.loc[c['CabinClass'] == 'UNKOWN', 'CabinCode'] = 99

    c['CabinNumber'] = c['Cabin'].apply(lambda x: len(x.split(' ')))

    c = c.drop(columns=['Cabin', 'CabinClass'])

    # Family size

    c['FamilySize'] = c['SibSp'] + c['Parch'] + 1

    c['IsAlone'] = 0

    c.loc[c['FamilySize'] == 1, 'IsAlone'] = 1

    c = c.drop(columns=['SibSp', 'Parch', 'FamilySize'])

    c = c.drop(columns=['PassengerId','Ticket'])

    return c
def train_and_predict(x, y):

    classifier = XGBClassifier()

    train_x, test_x, train_y, test_y = model_selection.train_test_split(x, y, test_size=0.25, random_state=42)

    classifier.fit(train_x, train_y)

    y_pred = classifier.predict(test_x)

    accuracy = metrics.accuracy_score(y_pred, test_y)

    print('Accuracy: {:.2f}'.format(accuracy * 100))

    return classifier
def train(x, y):

    classifier = XGBClassifier()

    classifier.fit(x, y)

    return classifier
df = pd.read_csv('../input/train.csv')

y = df['Survived']

df = df.drop(columns=['Survived'])

test_x = pd.read_csv('../input/test.csv')

print(test_x.isnull().any())

test_y = pd.read_csv('../input/gender_submission.csv')

df2 = feature_engineering(df)

print(df2.columns)

clssifier = train_and_predict(df2, y)

survived = clssifier.predict(feature_engineering(test_x))

submission = test_x.copy()

submission['Survived'] = survived

submission = submission.filter(items=['PassengerId', 'Survived'])

submission.to_csv('submission.csv', index = False)