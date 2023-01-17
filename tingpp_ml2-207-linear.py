# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read source data

raw = pd.read_csv(r'/kaggle/input/housesalesprediction/kc_house_data.csv')
# show the firs 5 rows for preview

raw.head()

# show the last 5 rows for preview

raw.tail()
# preview data type

raw.dtypes
# check if there is any null values in data source

count = 0

for each in raw.isnull():

    if each is True :

        count = count + 1

print('The null value count is : ' + str(count))
# preview price and sqft_living in scatter graph

plt.figure(figsize=(16, 8))

plt.scatter(raw['price'], raw['sqft_living'])

plt.xlabel("Price")

plt.ylabel("sqft_living")

plt.show()
# split the columns which I want to ML to test group and train group

X = raw['price'].values.reshape(-1,1)

y = raw['sqft_living'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 1)
# show the shape of each group

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# import module of linear regression then use it to fit the train group data.

from sklearn.linear_model import LinearRegression

model = LinearRegression()

output_model = model.fit(X_train, y_train)

output_model
# show the formula of linear regression / r squre value / intercept / slope

print('the linear model is: y = ', model.coef_[0][0], '* X + ', model.intercept_[0])

r_sq = model.score(X_train, y_train)

print('coefficient of determination(R Sq):', r_sq)

print('intercept : ', model.intercept_)

print('slope : ', model.coef_)
# use the model which applied by train group to predict test group

# show both scatter of test group and prediction line of test group

y_pre = model.predict(X_test)

plt.figure(figsize=(16, 8))

plt.scatter(X_test, y_test)

plt.plot(X_test, y_pre, c='black')

plt.xlabel("Price")

plt.ylabel("sqft_living")

plt.show()
# evaluation

from sklearn import metrics

MSE = metrics.mean_squared_error(y_test, y_pre)

RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))



print('MSE:',MSE)

print('RMSE:',RMSE)