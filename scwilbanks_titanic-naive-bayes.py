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

from sklearn.impute import SimpleImputer

from sklearn.naive_bayes import GaussianNB
train_df = pd.read_csv("/kaggle/input/titanic/train.csv", index_col='PassengerId')

test_df = pd.read_csv("/kaggle/input/titanic/test.csv", index_col='PassengerId')

X = pd.concat([train_df, test_df])
X
df.columns
df.describe(include='all')
def preprocess(X):

    

    

    # Feature Selection

    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    

    # Dummy Variables

    X = pd.get_dummies(X)

    

    # Imputation

    X = pd.DataFrame(data=SimpleImputer().fit_transform(X), columns=X.columns, index=X.index)

    

    # Split into train and test

    X_train = X.loc[:891]

    X_test = X.loc[892:]

    

    return X_train, X_test

    
y_train = X['Survived'].loc[:891]

X_train, X_test = preprocess(X)
gnb = GaussianNB()

gnb.fit(X_train, y_train)
y_hat = gnb.predict(X_test)
y_hat_df = pd.DataFrame({'PassengerId': X_test.index, 'Survived': y_hat})
y_hat_df.to_csv('naivebayes.csv', index=False)