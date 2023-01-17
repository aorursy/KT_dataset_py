# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_path = "../input/train_V2.csv"

train_data = pd.read_csv(train_path, index_col="Id")

train_data.head()
test_path = "../input/test_V2.csv"

test_data = pd.read_csv(test_path, index_col="Id")

test_data.head()
train_data.info()
columns = train_data.columns

for column in columns:

    print(f"{column}: {train_data[column].isna().sum()}")
data = train_data.copy()

data = data.dropna()
columns = data.columns

for column in columns:

    print(f"{column}: {data[column].isna().sum()}")
num_data = data.select_dtypes(include='number')
num_data.info()
# plt.figure(figsize=(15, 15))

# sns.heatmap(data=num_data.corr(), annot=True, fmt='.1f')

# plt.show()
# plt.figure(figsize=(15, 15))

# sns.scatterplot(x=num_data.walkDistance, y=num_data.winPlacePerc)

# plt.show()
# plt.figure(figsize=(15, 15))

# sns.scatterplot(x=num_data.damageDealt, y=num_data.kills)

# plt.show()
# average number of kills

print(data.kills.mean())
# maximum kills by a player

print(data.kills.max())
y = num_data.winPlacePerc

X = num_data.drop(columns='winPlacePerc')
test_X = test_data.select_dtypes(include='number')
from sklearn.tree import DecisionTreeRegressor



model = DecisionTreeRegressor()

model.fit(X, y)
output = model.predict(test_X)
out = pd.DataFrame({'Id': test_X.index, 'winPlacePerc': output})
out.to_csv('submission.csv', index=False)