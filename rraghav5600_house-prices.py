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
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import accuracy_score
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

t_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



ffile = t_df['Id']

df.drop(columns='Id', inplace=True)

t_df.drop(columns='Id', inplace=True)
df.head()
label = LabelEncoder()

cols = df.columns.tolist()

cols.remove('SalePrice')

for i in cols:

    try:

        t_df[i] = label.fit_transform(t_df[i].astype('str'))

        df[i] = label.fit_transform(df[i].astype('str'))

    except:

        pass

df.head()
X = df.drop(columns='SalePrice')

y = df['SalePrice']

test = t_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
rf = RandomForestClassifier(random_state=1, n_estimators=1000)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

accuracy_score(pred_rf, y_test)
mlp = MLPClassifier()

mlp.fit(X_train, y_train)

pred_mlp = mlp.predict(X_test)

accuracy_score(pred_mlp, y_test)
dtc = DecisionTreeRegressor(random_state=0)

dtc.fit(X_train, y_train)

pred_dtc = dtc.predict(X_test)

accuracy_score(pred_dtc, y_test)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='saga')

lr.fit(X_train, y_train)

pred = lr.predict(X_test)

accuracy_score(pred, y_test)
rf.fit(X,y)

vals_rf = rf.predict(test)

file = pd.DataFrame({'Id':ffile, 'SalePrice':vals_rf})

file.to_csv('submission_rf.csv', index = False)

file.head()
mlp.fit(X,y)

vals_mlp = mlp.predict(test)

file = pd.DataFrame({'Id':ffile, 'SalePrice':vals_mlp})

file.to_csv('submission_mlp.csv', index = False)

file.head()
dtc.fit(X,y)

vals_dtc = dtc.predict(test)

file = pd.DataFrame({'Id':ffile, 'SalePrice':vals_dtc})

file.to_csv('submission_dtc.csv', index = False)

file.head()
lr.fit(X,y)

vals_lr = lr.predict(test)

file = pd.DataFrame({'Id':ffile, 'SalePrice':vals_lr})

file.to_csv('submission_lr.csv', index = False)

file.head()