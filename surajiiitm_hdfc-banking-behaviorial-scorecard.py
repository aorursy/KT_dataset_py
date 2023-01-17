# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset_dir = "/kaggle/input/banking-beahviour-scorecard-for-customer/DataSet/"

train = pd.read_csv(dataset_dir + "Train.csv")

test = pd.read_csv(dataset_dir + "Test.csv")

submission = pd.read_csv(dataset_dir + "Sample_submission.csv")
train.head()
target = train["Col2"]

print(target.value_counts())

target.hist()
x_train = train.iloc[:, 2:]

y_train = train['Col2']

x_test = test.iloc[:, 1:]
def check_columns(df, column):

    for j in column:

        df[j] = df[j].replace('-', '0')

        df[j] = pd.to_numeric(df[j])

    return df
check_test_column = []

for column in x_test.columns:

    if (test[column].dtypes) != 'float64' and test[column].dtypes != 'int64':

        print(column, test[column].dtypes)

        check_test_column.append(column)
# checking if any column datatype is not numeric

for column in x_train.columns:

    if (x_train[column].dtypes) != 'float64' and x_train[column].dtypes != 'int64':

        print(column, x_train[column].dtypes)
object_column = ['Col747', 'Col836']

x_train = check_columns(x_train, object_column)
x_test = check_columns(x_test, check_test_column)

x_test.shape
x_train.fillna(value=0, inplace=True)

x_test.fillna(value=0, inplace=True)
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)



# # Dimensionality Reduction using PCA

# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)

# x_train = pca.fit_transform(x_train)

# x_test = pca.transform(x_test)

# explained_variance = pca.explained_variance_ratio_

# explained_variance



x_train.shape, x_test.shape
from xgboost import XGBClassifier

xgb_classifier = XGBClassifier(max_depth=5)

xgb_classifier.fit(x_train, y_train)
result = xgb_classifier.predict(x_test)

submission = pd.DataFrame({'Col1': test['Col1'], 'Col2': result})

submission.to_csv("xgboost_simple.csv", index=False)