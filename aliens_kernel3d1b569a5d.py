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
df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
df.dtypes
df.drop(columns = ['id'], inplace=True)

df.describe()
df['floors'].value_counts().to_frame()
import seaborn as sn



sn.boxplot(x=df['waterfront'], y=df['price'])

sn.regplot(x=df['sqft_above'], y=df['price'])
from sklearn.linear_model import LinearRegression



X = df['sqft_living'].values.reshape(-1, 1)

y = df['price'].values.reshape(-1, 1)



lr = LinearRegression()

lr.fit(X, y)

lr.score(X, y)
from sklearn.linear_model import LinearRegression



X = df[[

    'floors',

    'waterfront',

    'lat',

    'bedrooms',

    'sqft_basement',

    'view',

    'bathrooms',

    'sqft_living15',

    'sqft_above',

    'grade',

    'sqft_living']].values

y = df['price'].values.reshape(-1, 1)



lr = LinearRegression()

lr.fit(X, y)

lr.score(X, y)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, PolynomialFeatures



scaler = StandardScaler()

poly = PolynomialFeatures(2)



pipe = Pipeline(steps=[

    ('scale', scaler),

    ('poly', poly),

    ('lr', lr)

])

pipe.fit(X, y)

pipe.score(X, y)
from sklearn.linear_model import Ridge



ridge = Ridge(alpha=0.1)

ridge.fit(X, y)

ridge.score(X, y)
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



X_train_poly = PolynomialFeatures(2).fit_transform(X_train)

X_test_poly = PolynomialFeatures(2).fit_transform(X_test)



ridge = Ridge(alpha=0.1)

ridge.fit(X_train_poly, y_train)

ridge.score(X_test_poly, y_test)




