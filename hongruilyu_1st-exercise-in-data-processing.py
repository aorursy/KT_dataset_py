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
# First draft public in Kaggle. Just learn data science!

import matplotlib.pyplot as plt

from scipy import stats

import sklearn.linear_model as skl
data = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", header = 0, index_col = 0)

data.head()
"""Q1: What is the average of points of wines for each country? Ranking from top to bottom"""

data['points'].isnull().value_counts()
country_points_total = data.groupby(['country'])['points'].agg(sum)

country_points_number = data.groupby(['country'])['points'].count()

country_points = country_points_total / country_points_number
country_points.sort_values(ascending = False)
country_points.sort_values(ascending = False).plot.bar()

plt.show()
data_points = data['points']

data_price = data['price']

data_points_price = pd.DataFrame({"points":data_points, "price":data_price})

data_points_price = data_points_price.dropna(how = 'any')

data_points_price.corr()
reg = skl.LinearRegression()

reg.fit(data_points_price.price[:, np.newaxis], data_points_price.points)
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.scatter(data_points_price.price, data_points_price.points)

ax.plot(data_points_price.price, reg.predict(data_points_price.price[:, np.newaxis]), color = 'r')

plt.show()
"""Q3: Just curious about the price per points average for each country"""
data = data[['country', 'price', 'points']]

data = data.dropna(how = 'any')

data.head()
data_country_pp = data.groupby(by = ['country'])['price', 'points'].agg(sum)

data_country_numb = data.groupby(by = ['country'])['price'].count()

data_country_pp.price = data_country_pp.price / data_country_numb

data_country_pp.points = data_country_pp.points / data_country_numb

data_country_pp['point-price'] = data_country_pp.points / data_country_pp.price
data_country_pp[['point-price']]
data_country_pp[['point-price']].sort_values(by = 'point-price', ascending = False).plot.bar()

plt.show()