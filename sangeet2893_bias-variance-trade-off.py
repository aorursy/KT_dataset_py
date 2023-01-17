# Importing required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
ad  = pd.read_csv("../input/advertising.csv/Advertising.csv")
ad.drop("Unnamed: 0", axis=1, inplace=True)

ad.head()
# Fitting Linear Regression 

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(ad.TV.values.reshape(-1,1), ad.sales)
plt.figure(figsize=(8,5))

plt.subplot(111, facecolor='black')

plt.scatter(ad.TV, ad.sales, color='gold')

plt.plot(ad.TV,model.predict(ad.TV.values.reshape(-1,1)),color='red',linewidth=4)

plt.text(0,22,"Restrictive Model\nHigh Bias\nUnderfitting", color='white',size=15)

plt.text(220,1,"@DataScienceWithSan", color='white')
# Generating random data

np.random.seed(0)

x = np.random.normal(0,1,100)

y = 30 + 4*x - 2*(x**2) + 3*(x**3) + np.random.normal(0,1,100)

plt.figure(figsize=(12,4))



plt.subplot(121)

plt.ylim(10,60)

plt.xlim(-1,2)

plt.scatter(x,y,s=20,c='seagreen')

plt.plot(x, 27+12*x, color='orange', linewidth=1)



plt.subplot(122)

plt.ylim(10,60)

plt.xlim(-1,2)

plt.scatter(x,y,s=20,c='seagreen')

x2 = np.linspace(-3,3,50)

plt.plot(x2,30 + 4*x2 - 2*(x2**2) + 3*(x2**3), c='orange', linewidth=2)
from sklearn import neighbors

knn = neighbors.KNeighborsRegressor(n_neighbors=1, weights='distance')

knn.fit(ad.TV.values.reshape(-1,1), ad.sales)



x_points_ad = np.linspace(0,300,100)

y_knn_ad = knn.predict(x_points_ad.reshape(-1,1))



plt.figure(figsize=(8,5))

plt.subplot(111, facecolor='black')

plt.scatter(ad.TV, ad.sales, color='gold')

plt.plot(x_points_ad, y_knn_ad, color='red',linewidth=4)

plt.text(0,22,"Complex Model\nHigh Variance\nOverfitting", color='white', size=15)

plt.text(220,1,"@DataScienceWithSan", color='white')
## Making Random Data



np.random.seed(0)

x = np.random.normal(0,10,50)

y = 0.1*x + 0.01*(x**2) + 0.01*(x**3) + np.random.normal(0,10,50)



x = np.array(x).reshape(-1,1)

y = np.array(y).reshape(-1,1)
plt.figure(figsize=(14,8))



## Fitting and plotting Linear Regression

regression_model = LinearRegression()

regression_model.fit(x,y)

plt.subplot(221)

plt.scatter(x, y, edgecolor='skyblue',color='royalblue')

plt.plot(x,regression_model.predict(x),color='orange',linewidth=1)

plt.title("Figure 1 - Linear Regression")



#############################################################################



## Fitting and plotting polynomial regression

x_points = np.linspace(-25,25,100)

y2 =  0.1*x_points + 0.01*(x_points**2) + 0.01*(x_points**3)

plt.ylim(-150,150)

plt.subplot(222)

plt.scatter(x, y, edgecolor='skyblue',color='royalblue')

plt.plot(x_points,y2,color='orange',linewidth=2)

plt.title("Figure 2 - Polynomial Regression")



#############################################################################



## Fitting and plotting KNN with high k

from sklearn import neighbors

knn = neighbors.KNeighborsRegressor(n_neighbors=9, weights='distance')

knn.fit(x, y)

y_knn_h = knn.predict(x_points.reshape(-1,1))

plt.subplot(223)

plt.scatter(x, y, edgecolor='skyblue', color='royalblue')

plt.plot(x_points, y_knn_h, color='orange',linewidth=2)

plt.title("Figure 3 - KNN with 9 nearest neighbors")



#############################################################################



## Fitting and plotting KNN with k=1

from sklearn import neighbors

knn = neighbors.KNeighborsRegressor(n_neighbors=1, weights='distance')

knn.fit(x, y)

y_knn_h = knn.predict(x_points.reshape(-1,1))

#plt.subplot(111,facecolor='navy')

plt.subplot(224)

plt.scatter(x, y, edgecolor='skyblue',color='royalblue' )

plt.plot(x_points, y_knn_h, color='orange', linewidth=2)

plt.title("Figure 4 - KNN with 1 nearest neighbor")