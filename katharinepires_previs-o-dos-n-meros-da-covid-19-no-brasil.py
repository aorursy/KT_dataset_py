import numpy as np 

import matplotlib.pyplot as plt 

import pandas as pd

import seaborn as sns
covid = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df_BR = covid[covid['Country/Region'] == 'Brazil']

df_BR
sns.lineplot(df_BR['ObservationDate'], df_BR['Confirmed']);
sns.lineplot(df_BR['ObservationDate'], df_BR['Deaths']);
sns.lineplot(df_BR['ObservationDate'], df_BR['Recovered']);
df_sum = df_BR.groupby('ObservationDate').agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'}).reset_index()

plt.stackplot(df_sum['ObservationDate'], [df_sum['Confirmed'], df_sum['Deaths'], df_sum['Recovered']], labels = ['Confirmados', 'Mortos', 'Recuperados'])

plt.legend(loc = 'upper left')
confirmados = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

confirmados = confirmados[confirmados['Country/Region'] == 'Brazil']

confirmados
columns1 = confirmados.keys()

columns1
confirmados = confirmados.loc[:, columns1[4]:columns1[-1]]

confirmados
dates1 = confirmados.keys()

y_c = []

for i in dates1:

    y_c.append(confirmados[i].sum())
y_c = np.array(y_c).reshape(-1,1)

y_c 
x_c = np.arange(len(dates1)).reshape(-1,1)

x_c
forecast_c = np.arange(len(dates1) + 15).reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train_c, X_test_c, Y_train_c, Y_test_c = train_test_split(x_c, y_c, test_size = 0.40, shuffle = False)
from sklearn.preprocessing import PolynomialFeatures

poly_c = PolynomialFeatures(degree = 3)

X_train_poly_c = poly_c.fit_transform(X_train_c)

X_test_poly_c = poly_c.transform(X_test_c)
from sklearn.linear_model import LinearRegression

lr_c = LinearRegression()

lr_c.fit(X_train_poly_c, Y_train_c)
poly_pred_c = lr_c.predict(X_test_poly_c)

plt.plot(poly_pred_c, linestyle = 'dashed')

plt.plot(Y_test_c)
from sklearn.metrics import mean_absolute_error, mean_squared_error

print('MAE:', mean_absolute_error(poly_pred_c, Y_test_c))

print('MSE:', mean_squared_error(poly_pred_c, Y_test_c))

print('RMSE:', np.sqrt(mean_absolute_error(poly_pred_c, Y_test_c)))
mortes = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

mortes = mortes[mortes['Country/Region'] == 'Brazil']

mortes
columns = mortes.keys()

mortes = mortes.loc[:, columns[4]:columns[-1]]

mortes
dates = mortes.keys()

y = []

for i in dates:

    y.append(mortes[i].sum())
y = np.array(y).reshape(-1,1)

y
x = np.arange(len(dates)).reshape(-1,1)

x
forecast = np.arange(len(dates) + 15).reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.30, shuffle = False)
poly = PolynomialFeatures(degree = 2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)
lr = LinearRegression()

lr.fit(X_train_poly, Y_train)
poly_pred = lr.predict(X_test_poly)

plt.plot(poly_pred, linestyle = 'dashed')

plt.plot(Y_test)
print('MAE:', mean_absolute_error(poly_pred, Y_test))

print('MSE:', mean_squared_error(poly_pred, Y_test))

print('RMSE:', np.sqrt(mean_absolute_error(poly_pred, Y_test)))
recuperados = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

recuperados = recuperados[recuperados['Country/Region'] == 'Brazil']

recuperados
columns2 = recuperados.keys()

recuperados = recuperados.loc[:, columns2[4]:columns2[-1]]

recuperados
dates2 = recuperados.keys()

y_r = []

for i in dates2:

    y_r.append(recuperados[i].sum())
y_r = np.array(y_r).reshape(-1,1)

y_r
x_r = np.arange(len(dates2)).reshape(-1,1)

x_r
forecast_r = np.arange(len(dates2) + 15).reshape(-1,1)
X_train_r, X_test_r, Y_train_r, Y_test_r = train_test_split(x_r, y_r, test_size = 0.30, shuffle = False)
poly_r = PolynomialFeatures(degree = 3)

X_train_poly_r = poly_r.fit_transform(X_train_r)

X_test_poly_r = poly_r.transform(X_test_r)
lr_r = LinearRegression()

lr_r.fit(X_train_poly_r, Y_train_r)
poly_pred_r = lr_r.predict(X_test_poly_r)

plt.plot(poly_pred_r, linestyle = 'dashed')

plt.plot(Y_test_r)
print('MAE:', mean_absolute_error(poly_pred_r, Y_test_r))

print('MSE:', mean_squared_error(poly_pred_r, Y_test_r))

print('RMSE:', np.sqrt(mean_absolute_error(poly_pred_r, Y_test_r)))
X_train_all_c = poly_c.transform(forecast_c)

pred_all_c = lr_c.predict(X_train_all_c)



plt.plot(forecast_c[:-15], y_c, color='red')

plt.plot(forecast_c, pred_all_c, linestyle='dashed')

plt.title('Casos Confirmados de COVID-19')

plt.xlabel('Dias desde 22/01/2020')

plt.ylabel('Número de confirmados')

plt.legend(['Casos Confirmados', 'Previsões']);
X_train_all = poly.transform(forecast)

pred_all = lr.predict(X_train_all)



plt.plot(forecast[:-15], y, color='red')

plt.plot(forecast, pred_all, linestyle='dashed')

plt.title('Mortes por COVID-19')

plt.xlabel('Dias desde 22/01/2020')

plt.ylabel('Número de mortes')

plt.legend(['Mortes', 'Previsões']);
X_train_all_r = poly_r.transform(forecast_r)

pred_all_r = lr_r.predict(X_train_all_r)



plt.plot(forecast_r[:-15], y_r, color='red')

plt.plot(forecast_r, pred_all_r, linestyle='dashed')

plt.title('Recuperados da COVID-19')

plt.xlabel('Dias desde 22/01/2020')

plt.ylabel('Número de recuperados')

plt.legend(['Recuperados', 'Previsões']);