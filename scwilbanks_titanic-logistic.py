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
import numpy as np

import pandas as pd

import matplotlib as plt

from sklearn import linear_model

from sklearn import preprocessing

from sklearn.impute import SimpleImputer
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df
train_df.columns
dir(train_df)
train_df.describe(include='all')
def preprocess(df):

    

    # Feature Selection

    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    

    # Dummy Variables

    X = pd.get_dummies(X)

    

    # Imputation

    X = pd.DataFrame(data=SimpleImputer().fit_transform(X), columns=X.columns)

    

    # Scale

    X = pd.DataFrame(data=preprocessing.scale(X), columns=X.columns)

    

    return X

    
X_train = preprocess(train_df)



y = train_df['Survived']
reg = linear_model.LogisticRegression()

reg.fit(X_train, y)
X_test = preprocess(test_df)
y_hat = reg.predict(X_test)
y_hat_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_hat})
y_hat_df.to_csv('logistic.csv', index=False)