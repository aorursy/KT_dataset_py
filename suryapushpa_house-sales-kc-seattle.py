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
import pandas as pd 
pd.set_option('display.max_rows',None)

import numpy as np

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
data.head()
data.info()
data.isnull().sum()
corr = data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, linewidths=0.1, cmap='YlGnBu')
plt.show()
# Create x and y

featured_col = 'sqft_living'
x = data[featured_col]
y = data['price']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

x_train = x_train[:,np.newaxis]
y_train = y_train[:,np.newaxis]
x_test = x_test[:,np.newaxis]
y_test = y_test[:,np.newaxis]
reg = LinearRegression()
reg.fit(x_train, y_train)

reg.intercept_, reg.coef_
yi_val = reg.coef_ * 1000 + reg.intercept_
print(yi_val)
print(reg.predict([[1000]]))

mse = mean_squared_error(y_test, reg.predict(x_test))
print(np.sqrt(mse))
# Not great

print(reg.score(x_test, y_test))
# Our model is able to predict only 50% of the variability in house prices.


data['bedrooms'].value_counts().plot(kind='bar')
plt.xlabel("No. of bedrooms")
plt.ylabel("Count")
plt.show()
# Visualizing the locations

plt.figure(figsize=(10,10))
sns.jointplot(data['lat'], data['long'], size=10)
# How common factors are affecting the price of the houses ?

plt.scatter(data['bedrooms'], data['price'])
plt.show()
plt.scatter(data['bathrooms'], data['price'])
plt.show()
plt.scatter(data['sqft_living'], data['price'])
plt.show()
plt.scatter(data['sqft_lot'], data['price'])
plt.show()
plt.scatter(data['floors'], data['price'])
plt.show()
plt.scatter(data['waterfront'], data['price'])
plt.show()
plt.scatter(data['view'], data['price'])
plt.show()
plt.scatter(data['condition'], data['price'])
plt.show()
plt.scatter(data['grade'], data['price'])
plt.show()
plt.scatter(data['sqft_above'], data['price'])
plt.show()
plt.scatter(data['sqft_basement'], data['price'])
plt.show()
plt.scatter(data['yr_built'], data['price'])
plt.show()
plt.scatter(data['yr_renovated'], data['price'])
plt.show()
plt.scatter(data['zipcode'], data['price'])
plt.show()
plt.scatter(data['lat'], data['price'])
plt.show()
plt.scatter(data['long'], data['price'])
plt.show()

dates = [1 if '2014' in x else 0 for x in data['date']]
train_data = data
train_data['date'] = dates

train_data = train_data.drop(['id', 'price'], axis = 1)
train_data.head()
x_mtrain, x_mtest, y_mtrain, y_mtest = train_test_split(train_data, data['price'], test_size=0.10)
reg.fit(x_mtrain, y_mtrain)
y_mpredict = reg.predict(x_mtest)

mse = mean_squared_error(y_mtest, y_mpredict)
print(mse)

rms = reg.score(x_mtest, y_mtest)
print(rms)

# Only 72% which doesn't cross 85%, thus we need better model for our dataset
from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_split=2, learning_rate=0.03, loss='ls')
clf.fit(x_mtrain, y_mtrain)
y_mpredict = clf.predict(x_mtest)

mse = mean_squared_error(y_mtest, y_mpredict)
print(mse)

rms = clf.score(x_mtest, y_mtest)*100
print(rms)

# >87%, its amazing right!

