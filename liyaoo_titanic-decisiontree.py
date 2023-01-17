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
train_set = pd.read_csv('/kaggle/input/titanic/train.csv')

train_set.head()
train_set.info()
features = train_set[['Pclass', 'Age', 'Sex']]

label = train_set['Survived']

features.info()
features['Age'].fillna(features['Age'].mean(), inplace=True)

features['Sex'].replace('male', 0, inplace=True)

features['Sex'].replace('female', 1, inplace=True)

features.head()
from sklearn.model_selection import train_test_split

features_train, features_test, label_train, label_test = train_test_split(

    features, label, test_size=0.25, random_state=33)
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer(sparse=False)



features_train = vec.fit_transform(features_train.to_dict(orient='record'))

features_test = vec.fit_transform(features_test.to_dict(orient='record'))

print(vec.feature_names_)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(features_train, label_train)

label_predict = dtc.predict(features_test)

label_predict
from sklearn.metrics import classification_report

print(dtc.score(features_test, label_test))

print(classification_report(label_predict, label_test,

                            target_names=['died', 'survived']))