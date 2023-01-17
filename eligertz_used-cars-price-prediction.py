import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
vehicles_df = pd.read_csv('../input/craigslistVehicles.csv')
vehicles_df.info()
vehicles_df.head()
vehicles_df = vehicles_df.drop(columns=['city_url', 'image_url', 'lat', 'long'])
vehicles_df.shape
vehicles_df.drop_duplicates(subset='url')

vehicles_df.shape
vehicles_df.isnull().sum(axis=1).quantile(.95)
vehicles_df = vehicles_df[vehicles_df.isnull().sum(axis=1) < 9]

vehicles_df.shape
vehicles_df = vehicles_df[vehicles_df.price != 0]

vehicles_df.shape
plt.figure(figsize=(3,6))

sns.boxplot(y='price', data=vehicles_df);
vehicles_df = vehicles_df[vehicles_df.price < 100000]

vehicles_df.shape
plt.figure(figsize=(15,9))

ax = sns.countplot(x='year',data=vehicles_df);

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",fontsize=10);
vehicles_df = vehicles_df[vehicles_df.year > 1985]

vehicles_df.shape
vehicles_df.odometer.quantile(.999)
vehicles_df = vehicles_df[~(vehicles_df.odometer > 500000)]

vehicles_df.shape
plt.figure(figsize=(3,6))

sns.boxplot(y='odometer', data=vehicles_df);
vehicles_df.shape
sns.set(style="ticks", color_codes=True)

sns.pairplot(vehicles_df, hue="condition");
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import train_test_split as split

import warnings

from sys import modules
vehicles_df_to_learn = vehicles_df[['odometer','year','price']]
vehicles_df_to_learn = vehicles_df_to_learn.dropna()

vehicles_df_to_learn.shape
vehicles_df_train, vehicles_df_test = split(vehicles_df_to_learn, train_size=0.6, random_state=4222)
X_train = vehicles_df_train[['odometer','year']]

y_train = vehicles_df_train['price']
cars_lm = LinearRegression(fit_intercept=True)
cars_lm.fit(X_train, y_train)
print("The model intercept is: {}".format(cars_lm.intercept_))

print("The model coefficients are: {}".format(cars_lm.coef_[0]))
X_train['Price_prediction'] = cars_lm.predict(X_train)

X_train.head()
cars_train_rmse = np.sqrt(MSE(y_train, X_train['Price_prediction']))

print("RMSE = {:.2f}".format(cars_train_rmse))
cars_lm_test = LinearRegression()
X_test = vehicles_df_test[['odometer','year']]

y_test = vehicles_df_test['price']
cars_lm_test.fit(X_test, y_test)
X_test['price_prediction'] = cars_lm_test.predict(X_test)

X_test.head()
cars_test_rmse = np.sqrt(MSE(y_test, X_test['price_prediction']))

print("RMSE = {:.2f}".format(cars_test_rmse))
vehicles_df_to_learn2 = vehicles_df[['odometer','year','price', 'transmission', 'title_status', 'condition']]
vehicles_df_to_learn2.info()
vehicles_df_to_learn2 = vehicles_df[['odometer','year','price', 'transmission', 'title_status']]

vehicles_df_to_learn2 = vehicles_df_to_learn2.dropna()

vehicles_df_to_learn2.shape
vehicles_df_to_learn2.head()
vehicles_df_to_learn2['transmission_automatic'] = vehicles_df_to_learn2['transmission'].apply(lambda x: 1 if x == 'automatic' else 0)

vehicles_df_to_learn2['transmission_manual'] = vehicles_df_to_learn2['transmission'].apply(lambda x: 1 if x == 'manual' else 0)

vehicles_df_to_learn2['transmission_other'] = vehicles_df_to_learn2['transmission'].apply(lambda x: 1 if x == 'other' else 0)
vehicles_df_to_learn2 = vehicles_df_to_learn2.reset_index()

vehicles_df_to_learn2.head()
dum = pd.get_dummies(vehicles_df_to_learn2['title_status']).reset_index()
dum.head()
vehicles_df_to_learn2 = pd.merge(vehicles_df_to_learn2, dum, on='index')

vehicles_df_to_learn2 = vehicles_df_to_learn2.drop(columns=['index', 'transmission', 'title_status'])
vehicles_df_to_learn2.head()
vehicles_df_train2, vehicles_df_test2 = split(vehicles_df_to_learn2, train_size=0.6, random_state=4222)

X_train2 = vehicles_df_train2[['odometer','year', 'transmission_automatic', 'transmission_manual', 'transmission_other', 'clean', 'lien', 'missing', 'parts only', 'rebuilt', 'salvage']]

y_train2 = vehicles_df_train2['price']

cars_lm2 = LinearRegression(fit_intercept=True)

cars_lm2.fit(X_train2, y_train2)
print("The model intercept is: {}".format(cars_lm2.intercept_))

print("The model coefficients are: {}".format(cars_lm2.coef_[0]))

X_train2['Price_prediction'] = cars_lm2.predict(X_train2)

cars_train_rmse2 = np.sqrt(MSE(y_train2, X_train2['Price_prediction']))

print("RMSE = {:.2f}".format(cars_train_rmse2))
cars_lm_test2 = LinearRegression()

X_test2 = vehicles_df_test2[['odometer','year', 'transmission_automatic', 'transmission_manual', 'transmission_other', 'clean', 'lien', 'missing', 'parts only', 'rebuilt', 'salvage']]

y_test2 = vehicles_df_test2['price']

cars_lm_test2.fit(X_test2, y_test2)

X_test2['price_prediction'] = cars_lm_test2.predict(X_test2)

X_test2.head()

cars_test_rmse2 = np.sqrt(MSE(y_test2, X_test2['price_prediction']))

print("RMSE = {:.2f}".format(cars_test_rmse2))
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import pairwise_distances

from sklearn import neighbors

from math import sqrt

from sklearn.metrics import mean_squared_error 
vehicles_df_knn_train, vehicles_df_knn_test = split(vehicles_df_to_learn, train_size=0.6, random_state=4222)

X_first = vehicles_df_knn_train.drop('price', axis=1)

y_first = vehicles_df_knn_train['price']



X_second = vehicles_df_knn_test.drop('price', axis=1)

y_second = vehicles_df_knn_test['price']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))



X_first_scaled = scaler.fit_transform(X_first)

X_first = pd.DataFrame(X_first_scaled)



X_second_scaled = scaler.fit_transform(X_second)

X_second = pd.DataFrame(X_second_scaled)
rmse_val2 = [] #to store rmse values for different k

for K in range(20):

    K += 1

    model = neighbors.KNeighborsRegressor(n_neighbors = K)



    model.fit(X_first, y_first)  #fit the model

    pred=model.predict(X_second) #make prediction on test set

    error = sqrt(mean_squared_error(y_second, pred)) #calculate rmse

    rmse_val2.append(error) #store rmse values

    print('RMSE value for k= ' , K , 'is:', error)
#plotting the rmse values against k values

curve = pd.DataFrame(rmse_val2) #elbow curve 

curve.plot()
vehicles_df_to_learn2.head()
vehicles_df_knn_train2, vehicles_df_knn_test2 = split(vehicles_df_to_learn2, train_size=0.6, random_state=4222)

X_first2 = vehicles_df_knn_train2.drop('price', axis=1)

y_first2 = vehicles_df_knn_train2['price']



X_second2 = vehicles_df_knn_test2.drop('price', axis=1)

y_second2 = vehicles_df_knn_test2['price']
scaler = MinMaxScaler(feature_range=(0, 1))



X_first_scaled2 = scaler.fit_transform(X_first2)

X_first2 = pd.DataFrame(X_first_scaled2)



X_second_scaled2 = scaler.fit_transform(X_second2)

X_second2 = pd.DataFrame(X_second_scaled2)
rmse_val3 = [] 

K = 2

for i in range(5):

    K += 1

    model2 = neighbors.KNeighborsRegressor(n_neighbors = K)

    model2.fit(X_first2, y_first2)  

    pred2=model2.predict(X_second2) 

    error2 = sqrt(mean_squared_error(y_second2, pred2)) 

    rmse_val3.append(error2) 

    print('RMSE value for k= ' , K , 'is:', error2)