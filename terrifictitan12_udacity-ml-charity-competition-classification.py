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
train = pd.read_csv("/kaggle/input/udacity-mlcharity-competition/census.csv")
test = pd.read_csv("/kaggle/input/udacity-mlcharity-competition/test_census.csv")
train.head()
train['native-country'].unique()
mapp_1 = {' State-gov' : 0, ' Self-emp-not-inc' : 1, ' Private' : 2, ' Federal-gov' : 3,' Local-gov' : 4,
          ' Self-emp-inc' : 5, ' Without-pay' : 6}

mapp_2 = {' Bachelors' : 0, ' HS-grad' : 1, ' 11th' : 2, ' Masters' : 3, ' 9th' : 4,' Some-college' : 5, 
          ' Assoc-acdm' : 6, ' 7th-8th' : 7, ' Doctorate' : 8,' Assoc-voc' : 9, ' Prof-school' : 10, 
          ' 5th-6th' : 11, ' 10th' : 12, ' Preschool' : 13,' 12th' : 14, ' 1st-4th' : 15}

mapp_3 = {' Never-married' : 0, ' Married-civ-spouse' : 1, ' Divorced' : 2,' Married-spouse-absent' : 3,
          ' Separated' : 4, ' Married-AF-spouse' : 5,' Widowed' : 5}

mapp_4 = {' Adm-clerical' : 0, ' Exec-managerial' : 1, ' Handlers-cleaners' : 2,
       ' Prof-specialty' : 3, ' Other-service' : 4, ' Sales' : 5, ' Transport-moving' : 6,
       ' Farming-fishing' : 7, ' Machine-op-inspct' : 8, ' Tech-support' : 9,
       ' Craft-repair' : 10, ' Protective-serv' : 11, ' Armed-Forces' : 12,' Priv-house-serv' : 13}

mapp_5 = {' Not-in-family' : 0, ' Husband' : 1, ' Wife' : 2, ' Own-child' : 3, ' Unmarried' : 4,' Other-relative' : 5}

mapp_6 = {' White' : 0, ' Black' : 1, ' Asian-Pac-Islander' : 2, ' Amer-Indian-Eskimo' : 3, ' Other' : 4}

mapp_7 = {' Male' : 0, ' Female' : 1}
train['workclass'] = train['workclass'].map(mapp_1)
train['education_level'] = train['education_level'].map(mapp_2)
train['marital-status'] = train['marital-status'].map(mapp_3)
train['occupation'] = train['occupation'].map(mapp_4)
train['relationship'] = train['relationship'].map(mapp_5)
train['race'] = train['race'].map(mapp_6)
train['sex'] = train['sex'].map(mapp_7)


test['workclass'] = test['workclass'].map(mapp_1)
test['education_level'] = test['education_level'].map(mapp_2)
test['marital-status'] = test['marital-status'].map(mapp_3)
test['occupation'] = test['occupation'].map(mapp_4)
test['relationship'] = test['relationship'].map(mapp_5)
test['race'] = test['race'].map(mapp_6)
test['sex'] = test['sex'].map(mapp_7)
test.head()
from sklearn.preprocessing import LabelEncoder
test.isnull().sum()
test.dropna(inplace=True)
len(test)
le = LabelEncoder()
train['native-country'] = le.fit_transform(train['native-country'])
test['native-country'] = le.fit_transform(test['native-country'])
train.dtypes
test.dtypes
train['income'].unique()
mapp_income = {'<=50K' : 0, '>50K' : 1}
train['income'] = train['income'].map(mapp_income)
x_train = train.drop('income', axis=1)
y_train = train['income']

x_test = test
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
pred
x_train
model.predict([[28, 2, 0, 13.0, 1, 3, 2, 1, 1, 0.0, 0.0, 40.0, 4]])
