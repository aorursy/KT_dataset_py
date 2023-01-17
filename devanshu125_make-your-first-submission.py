# import relevant libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# Used to display plots inside this notebook, do not use plt.show() with this.

%matplotlib inline
# read data

train = pd.read_csv('../input/now-mozilla-club-mpstme/train.csv')

test = pd.read_csv('../input/now-mozilla-club-mpstme/test.csv')

sample_sub = pd.read_csv('../input/now-mozilla-club-mpstme/sample_sub.csv')
# print shape

train.shape, test.shape
# print first 5 rows

train.head()
# check for missing values

train.isna().sum()
def fill_missing(data):

    data['Trust'] = data['Trust'].fillna(data['Trust'].mean())
fill_missing(train)

fill_missing(test)
train.info()
test.info()
# split the data into X and y. X will include all the features and y will be our target variable.

X = train.drop(['Country', 'Happiness Score'], axis=1)

y = train['Happiness Score']
from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(X, y)
# store the predictions in a variable

preds = lr.predict(test.drop(['Country'], axis=1))
# print the preds

preds[0:5]
mysub = sample_sub.copy()
# replace the values in Happiness Score column with your happiness score

mysub['Happiness Score'] = preds
# print sample sub

mysub.head()
# convert your file to csv

mysub.to_csv('submission.csv', index=False)