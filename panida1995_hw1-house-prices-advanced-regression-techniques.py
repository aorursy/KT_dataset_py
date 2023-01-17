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

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df = df[['MSSubClass', 'MSZoning','LotFrontage','LotArea','SalePrice']]

print(df)





from  sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categories='auto')

feature_arr = ohe.fit_transform(df[['MSZoning']]).toarray()

feature_labels = ohe.categories_

feature_labels = np.array(feature_labels).ravel()

features = pd.DataFrame(feature_arr, columns=feature_labels)

print(features)
del df['MSZoning']

df = pd.concat([features,df], axis=1, sort=False)



df = df.dropna()

df.isnull().sum().sum()

df.head()

print(df)

train_input = df.iloc[:, :-1].values

train_output = df.iloc[:,8].values

print(train_output)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_input, train_output, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print(y_pred)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred)  