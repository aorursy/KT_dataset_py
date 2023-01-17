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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import pylab as pl

import matplotlib.pyplot as plt

%matplotlib inline

data=pd.read_csv("../input/real-estate-dataset/data.csv")
data.head()
data.isnull().any()
data.fillna(data.mean(), inplace=True)
data.isnull().any()
data.corr() 
plt.subplots(figsize=(12,7))

sns.heatmap(data.corr(), annot=True, cmap='YlGnBu', linewidths=1 );
plt.scatter(data.CRIM, data.MEDV, color='green', alpha=0.8)

plt.xlabel("Criminal")

plt.ylabel("Value")

plt.show()
house = np.random.rand(len(data)) < 0.8

train = data[house]

test = data[~house]
plt.scatter(train.CRIM, train.MEDV,  color='blue')

plt.xlabel("Criminal")

plt.ylabel("Value")

plt.show()



from sklearn import linear_model

regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['CRIM']])

train_y = np.asanyarray(train[['MEDV']])

regr.fit (train_x, train_y)

# The coefficients

print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)

plt.scatter(train.CRIM, train.MEDV,  color='blue')

plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')

plt.xlabel("Criminal")

plt.ylabel("Value")



from sklearn.metrics import r2_score



test_x = np.asanyarray(test[['CRIM']])

test_y = np.asanyarray(test[['MEDV']])

test_y_hat = regr.predict(test_x)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
plt.scatter(data.RM, data.MEDV, color='purple', alpha=0.8)

plt.xlabel("Room")

plt.ylabel("Value")

plt.show()
plt.scatter(train.RM, train.MEDV,  color='blue')

plt.xlabel("Room")

plt.ylabel("Value")

plt.show()



regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['RM']])

train_y = np.asanyarray(train[['MEDV']])

regr.fit (train_x, train_y)

# The coefficients

print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)
plt.scatter(train.RM, train.MEDV,  color='blue')

plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')

plt.xlabel("Room")

plt.ylabel("Value")





test_x = np.asanyarray(test[['RM']])

test_y = np.asanyarray(test[['MEDV']])

test_y_hat = regr.predict(test_x)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs 

from sklearn.preprocessing import StandardScaler

dt = data[['CRIM','RM', 'MEDV']]

X = dt.values[:,1:]

X = np.nan_to_num(X)

Clus_dataSet = StandardScaler().fit_transform(X)

Clus_dataSet
clusterNum = 3

k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)

k_means.fit(X)

labels = k_means.labels_

print(labels)
dt["data_km"] = labels

dt.head(5)
dt.groupby('data_km').mean()
area = np.pi * ( X[:, 1])**2  

plt.scatter(X[:, 0], X[:, 1], s=area, c=labels.astype(np.float), alpha=0.5)

plt.xlabel('Room', fontsize=18)

plt.ylabel('Value', fontsize=16)



plt.show()
from mpl_toolkits.mplot3d import Axes3D 

fig = plt.figure(1, figsize=(8, 6))

plt.clf()

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)



plt.cla()

# plt.ylabel('Age', fontsize=18)

# plt.xlabel('Income', fontsize=16)

# plt.zlabel('Education', fontsize=16)

ax.set_xlabel('Criminal')

ax.set_ylabel('Room')

ax.set_zlabel('Value')



ax.scatter(X[:, 1], X[:, 0], X[:, 1], c= labels.astype(np.float))