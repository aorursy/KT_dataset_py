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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/irish-weather-hourly-data/hourly_irish_weather.csv')

df
stations = df['station'].unique()

stations
type(df['date'][0])
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)

type(df['date'][0])
df = df.set_index('date')

df
plt.figure(figsize=(16, 9))

sns.boxplot(x="temp", y="station", data=df,

            whis="range", palette="vlag")
station3 = df.loc[lambda df: df.station == stations[3], :]

station3
sns.set()

%matplotlib inline

plt.figure(figsize=(16, 9))

sns.lineplot(x=station3.index,y="temp", data=station3, hue='station')
station3['month_str'] = station3.index.strftime('%b')

station3['month_num'] = station3.index.month

station3
plt.figure(figsize=(16, 9))

sns.boxplot(x="temp", y="month_str", data=station3, palette="vlag")
station3.columns
station3 = station3[['Unnamed: 0', 'station', 'county', 'longitude', 'latitude', 'month_str', 'month_num', 'rain',

       'temp', 'wetb', 'dewpt', 'vappr', 'rhum', 'msl', 'wdsp', 'wddir', 'ww',

       'w', 'sun', 'vis', 'clht', 'clamt']]
station3
station3.loc[:, 'rain':'wddir'].describe()
station3.loc[:, 'month_str':'wddir'].corr()
sns.pairplot(station3.loc[:, 'month_str':'wddir'], vars=['rain', 'temp', 'wetb', 'dewpt', 'vappr'],  hue='month_str', height=4)
station3['year'] = station3.index.year

station3['day'] = station3.index.day

station3['hour'] = station3.index.hour



station3
station3 = station3[['Unnamed: 0', 'station', 'county', 'longitude', 'latitude', 'month_str', 'year',

       'month_num', 'day', 'hour', 'rain', 'temp', 'wetb', 'dewpt', 'vappr', 'rhum', 'msl',

       'wdsp', 'wddir', 'ww', 'w', 'sun', 'vis', 'clht', 'clamt']]

station3
station3 = station3.dropna(subset=['temp'])
from sklearn.model_selection import train_test_split



X = station3.month_num

X = np.array(X)

X = X.reshape(-1,1)

X = np.append(X, np.array(station3.day).reshape(-1,1), axis=1)

X = np.append(X, np.array(station3.hour).reshape(-1,1), axis=1)

X.shape
X
y = np.array(station3.temp)

y
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train
from sklearn.linear_model import SGDRegressor



sgd_reg = SGDRegressor()

sgd_reg.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

sgd_reg_predict = sgd_reg.predict(X_train)

sgd_mse = mean_squared_error(y_train, sgd_reg_predict)

print("RMSE Entrenamiento: ", np.sqrt(sgd_mse))
sgd_reg_predict = sgd_reg.predict(X_test)

sgd_mse = mean_squared_error(y_test, sgd_reg_predict)

print("RMSE Test: ", np.sqrt(sgd_mse))
pred = np.array([11, 28, 14]).reshape(1,-1)

sgd_reg.predict(pred)
from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(degree=4)

poly = poly.fit_transform(X_train)
from sklearn.linear_model import LinearRegression 

lin_reg = LinearRegression()

lin_reg.fit(poly, y_train)
lin_reg_predict = lin_reg.predict(poly)

lin_mse = mean_squared_error(y_train, lin_reg_predict)

print("RMSE Entrenamiento: ", np.sqrt(lin_mse))
poly_test = PolynomialFeatures(degree=4)

poly_test = poly_test.fit_transform(X_test)

lin_reg_predict = lin_reg.predict(poly_test)

lin_mse = mean_squared_error(y_test, lin_reg_predict)

print("RMSE Test: ", np.sqrt(lin_mse))
pred_lin = np.array([11, 28, 16]).reshape(1,-1)

pred_poly = PolynomialFeatures(degree=4)

pred_poly = pred_poly.fit_transform(pred_lin)

lin_reg.predict(pred_poly)
from sklearn.model_selection import cross_val_score



lin_reg2 = LinearRegression()

poly_total = PolynomialFeatures(degree=4)

poly_total = poly_total.fit_transform(X)



scores = cross_val_score(lin_reg2, poly_total, y, scoring="neg_mean_squared_error", cv=100)

rmse_scores = np.sqrt(-scores)
print("Scores: ", rmse_scores)

print("Promedio: ", rmse_scores.mean())

print("Desvío estandar: ", rmse_scores.std())
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], 'max_iter': [1000, 1500, 2000, 2500],

     'penalty': ['none', 'l2', 'l1', 'elasticnet'], 'validation_fraction': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},

]
sgd_reg2 = SGDRegressor()

grid_search = GridSearchCV(sgd_reg2, param_grid, cv=5,

                           scoring='neg_mean_squared_error', 

                           return_train_score=True)

grid_search.fit(X_train, y_train)
sgd_reg3 = SGDRegressor(alpha=0.0001, average=False,

                                    early_stopping=False, epsilon=0.1,

                                    eta0=0.01, fit_intercept=True,

                                    l1_ratio=0.15, learning_rate='invscaling',

                                    loss='squared_loss', max_iter=1000,

                                    n_iter_no_change=5, penalty='l2',

                                    power_t=0.25, random_state=None,

                                    shuffle=True, tol=0.001,

                                    validation_fraction=0.1,

                                    warm_start=False)

sgd_reg3.fit(X_train, y_train)
sgd_reg_predict3 = sgd_reg3.predict(X_train)

sgd_mse3 = mean_squared_error(y_train, sgd_reg_predict3)

print("RMSE Entrenamiento: ", np.sqrt(sgd_mse3))
sgd_reg_predict3 = sgd_reg3.predict(X_test)

sgd_mse3 = mean_squared_error(y_test, sgd_reg_predict3)

print("RMSE Test with Grid search: ", np.sqrt(sgd_mse3))

print("RMSE Test without Grid search: ", np.sqrt(sgd_mse))
param_grid2 = [

    {'fit_intercept': [True, False], 'normalize': [True, False],

     'copy_X': [True, False], 'n_jobs': [1, -1]},

]



lin_reg3 = LinearRegression()

grid_search2 = GridSearchCV(lin_reg3, param_grid2, cv=5,

                           scoring='neg_mean_squared_error', 

                           return_train_score=True)

grid_search2.fit(poly, y_train)
lin_reg4 = LinearRegression(copy_X=True, fit_intercept=True,

                                        n_jobs=None, normalize=False)

lin_reg4 = lin_reg4.fit(poly, y_train)

lin_reg_predict4 = lin_reg4.predict(poly)

lin_mse4 = mean_squared_error(y_train, lin_reg_predict4)

print("RMSE Entrenamiento: ", np.sqrt(lin_mse4))
lin_reg_predict4 = lin_reg4.predict(poly_test)

lin_mse4 = mean_squared_error(y_test, lin_reg_predict4)

print("RMSE Test: ", np.sqrt(lin_mse4))