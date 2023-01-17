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
#Load data

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

train['Embarked'] = train['Embarked'].fillna('S')

test['Embarked'] = test['Embarked'].fillna('S')
train['family'] = train['SibSp'] + train['Parch']

test['family'] = test['SibSp'] + test['Parch']

def add_dummy(df):

    df['Pclass'] = df['Pclass'].astype(np.str)

    temp = pd.get_dummies(df[['Sex','Embarked','Pclass']], drop_first = False)

    temp['PassengerId'] = df['PassengerId']

    return pd.merge(df, temp)

train = add_dummy(train)

test = add_dummy(test)
def get_feature_mat(df):

    temp = df.drop(columns=['PassengerId','Name','Sex','SibSp','Parch','Ticket','Embarked','Age','Cabin','Pclass'])

    try:

        temp = temp.drop(columns=['Survived'])

    except:

        pass

    return temp.as_matrix()

x_train = get_feature_mat(train)

x_test = get_feature_mat(test)
y_train = train['Survived'].as_matrix()

from xgboost import XGBClassifier

xgb = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)

xgb.fit(x_train, y_train)

y_test_xgb = xgb.predict(x_test)
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=+9), 'JST')

ts = datetime.now(JST).strftime('%y%m%d%H%M')



test["Survived"] = y_test_xgb.astype(np.int)

test[["PassengerId","Survived"]].to_csv(('submit_'+ts+'.csv'),index=False)