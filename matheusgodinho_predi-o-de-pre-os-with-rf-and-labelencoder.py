# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from fastai import *

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv', header=0)

df = df.drop(columns=['sale_id', 'address', 'apartment_number', 'ease-ment'])
df.land_square_feet = pd.to_numeric(df.land_square_feet, errors='coerce').fillna(0)

df.gross_square_feet = pd.to_numeric(df.gross_square_feet, errors='coerce').fillna(0)
df.sale_date = pd.to_datetime(df.sale_date).dt.strftime("%Y%m%d")

# Categorical boolean mask

categorical_feature_mask = df.dtypes==object

# filter categorical columns using mask and turn it into a list

categorical_cols = df.columns[categorical_feature_mask].tolist()

categorical_cols
# import labelencoder

from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder object

le = LabelEncoder()

df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))

df.head(10)
def print_score(m):

    res = [m.score(X_train, y_train), m.score(X_test, y_test)]

    print(res)
from sklearn.model_selection import train_test_split  

y = df['sale_price'].values

X = df.drop(columns=['sale_price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor



m = RandomForestRegressor(n_estimators=100, bootstrap=False, n_jobs=-1, max_leaf_nodes=100)

m.fit(X_train, y_train)

print_score(m)
valid = pd.read_csv('../input/valid.csv', header=0, low_memory=False, parse_dates=["sale_date"])

test = pd.read_csv('../input/test.csv', header=0, low_memory=False, parse_dates=["sale_date"])
result = pd.DataFrame(columns=['sale_id', 'sale_price'])



result.sale_id = np.append(valid.sale_id.values, test.sale_id.values)
valid = valid.drop(columns=['sale_id', 'address', 'apartment_number', 'ease-ment'])

valid.land_square_feet = pd.to_numeric(valid.land_square_feet, errors='coerce').fillna(0)

valid.gross_square_feet = pd.to_numeric(valid.gross_square_feet, errors='coerce').fillna(0)

valid.sale_date = pd.to_datetime(valid.sale_date).dt.strftime("%Y%m%d")

valid[categorical_cols] = valid[categorical_cols].apply(lambda col: le.fit_transform(col))

predictionValid = m.predict(valid)
test = test.drop(columns=['sale_id', 'address', 'apartment_number', 'ease-ment'])

test.land_square_feet = pd.to_numeric(test.land_square_feet, errors='coerce').fillna(0)

test.gross_square_feet = pd.to_numeric(test.gross_square_feet, errors='coerce').fillna(0)

test.sale_date = pd.to_datetime(test.sale_date).dt.strftime("%Y%m%d")

test[categorical_cols] = test[categorical_cols].apply(lambda col: le.fit_transform(col))

predictionTest = m.predict(test)
predictions = np.append(predictionValid, predictionTest)

result.sale_price = predictions

result.info()
result.to_csv('result.csv')