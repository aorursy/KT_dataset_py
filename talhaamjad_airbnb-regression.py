# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df
# Print the shape of the data 
# data = data.sample(frac = 0.1, random_state = 48) 
print(df.shape) 
print(df.describe()) 
df.head()
cdf = df[['price','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]
cdf.head(9)
viz = cdf[['price','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]
viz.hist()
plt.show()
plt.scatter(cdf.price, cdf.number_of_reviews,  color='blue')
plt.xlabel("number_of_reviews")
plt.ylabel("price")
plt.show()
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
plt.scatter(train.price, train.number_of_reviews,  color='blue')
plt.xlabel("price")
plt.ylabel("number_of_reviews")
plt.show()
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['price']])
train_y = np.asanyarray(train[['number_of_reviews']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
plt.scatter(train.price, train.number_of_reviews,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("price")
plt.ylabel("number_of_reviews")
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['price']])
test_y = np.asanyarray(test[['number_of_reviews']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )