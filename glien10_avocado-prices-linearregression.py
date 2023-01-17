import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
path = '/kaggle/input/avocado.csv'

df = pd.read_csv(path, keep_default_na=False)
df.sort_values(by=['Date'], inplace=True)
df.head(5)
# Check if there are any missing values. If we find missing values, we expect to see two types of outputs - True and False

df.isnull().any().describe()
def get_weighted_average(arr1, arr2):

    s1 = np.dot(arr1, arr2)

    s2 = sum(arr2)

    return s1 / s2
f1 = ['Date', 'AveragePrice', 'Total Volume']

dates = df.Date.unique()



arr = []

for date in dates:

    temp = df[df.Date == date].copy()

    avgPrices = temp['AveragePrice']

    totalVolume = temp['Total Volume']

    weightedAvg = get_weighted_average(avgPrices, totalVolume)

    totalVolumeDay = sum(totalVolume)

    arr.append([date, weightedAvg, totalVolumeDay])
headers = ['date', 'weightedAvgPrice', 'totalVolume']

df1 = pd.DataFrame(data=arr, columns=headers)
df1.head(5)
X = np.array(df1['totalVolume']).reshape(-1, 1)

y = df1['weightedAvgPrice']
clf = LinearRegression(fit_intercept=True)

clf.fit(X, y)
y_pred = clf.predict(X)
score = mean_squared_error(y, y_pred)

score
plt.title('Avocado Trending Prices')

plt.xlabel('Volume')

plt.ylabel('Prices')

plt.scatter(X, y, color='black')

plt.plot(X, y_pred, color='blue')

plt.xticks(())

plt.yticks(())

plt.show()
df2 = df1.copy()

df2.drop(columns=['date'], inplace=True)

df2['idx'] = [ i for i in range(len(df2)) ]
df2.head(5)
X1 = np.array(df2['idx']).reshape(-1, 1)

y1 = df1['weightedAvgPrice']

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.25)
clf1 = LinearRegression(fit_intercept=True)

clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
score = mean_squared_error(y_test, y_pred)

score
plt.title('Avocado Trending Prices')

plt.xlabel('Time')

plt.ylabel('Prices')

plt.scatter(X_test, y_test, color='black')

plt.plot(X_test, y_pred, color='blue')

plt.xticks(())

plt.yticks(())

plt.show()