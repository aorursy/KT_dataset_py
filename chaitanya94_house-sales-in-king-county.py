# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/kc_house_data.csv')

df['date'] = pd.to_datetime(df['date'])

df['date'] = (df['date'] - df['date'].min())  / np.timedelta64(1,'D')

#print(df.head())
# Visualization -73

labels = ['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

y = 'price'

for i in range(len(labels)):

    plt.figure()

    plt.scatter(df[labels[i]], df[y])

    plt.xlabel(labels[i])

    plt.ylabel(y)
#df.info()

train = df[:20000]

test = df[20000:-1]

#test.info()

y = train['price']

y_test = test['price']

train = train.drop('price', 1)

test = test.drop('price', 1)
# Linear Regression

from sklearn import linear_model

regr = linear_model.LinearRegression()

regr.fit(train, y)

y_pred = regr.predict(test)

mse = sum((y_pred - y_test)**2)/len(y_pred)

print(mse)