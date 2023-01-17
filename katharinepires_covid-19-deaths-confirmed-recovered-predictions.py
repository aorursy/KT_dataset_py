import numpy as np 

import matplotlib.pyplot as plt 

import pandas as pd

import seaborn as sns
covid = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

covid
df_BR = covid[covid['Country/Region'] == 'Brazil']

df_BR
sns.lineplot(df_BR['ObservationDate'], df_BR['Confirmed']);
df_sum = covid.groupby('ObservationDate').agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'}).reset_index()

df_sum
plt.stackplot(df_sum['ObservationDate'], [df_sum['Confirmed'], df_sum['Deaths'], df_sum['Recovered']],

              labels = ['Confirmed', 'Deaths', 'Recovered'])

plt.legend(loc = 'upper left')
sns.pairplot(covid)
covid_line_list = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

covid_line_list
sns.distplot(covid_line_list['age'])
deaths = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

deaths
deaths[deaths['Country/Region'] == 'Brazil']
columns = deaths.keys()

columns
deaths = deaths.loc[:, columns[4]:columns[-1]]

deaths
len(deaths.keys())
deaths['1/22/20'].sum()
deaths['8/12/20'].sum()
#creating the variable:

dates = deaths.keys()

y = []

for i in dates:

    y.append(deaths[i].sum())
print(y)
y = np.array(y).reshape(-1,1) #will be transformed into matrix format

y
x = np.arange(len(dates)).reshape(-1,1) #will generate the number of dates according to the size of the "dates"

x
forecast = np.arange(len(dates) + 15).reshape(-1,1)

forecast
x.shape, y.shape, forecast.shape 
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.10, shuffle = False)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)
X_train.shape, X_test.shape, X_train_poly.shape, X_test_poly.shape
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train_poly, Y_train)
poly_pred = lr.predict(X_test_poly)

plt.plot(poly_pred, linestyle = 'dashed')

plt.plot(Y_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error

print('MAE:', mean_absolute_error(poly_pred, Y_test))

print('MSE:', mean_squared_error(poly_pred, Y_test))

print('RMSE:', np.sqrt(mean_absolute_error(poly_pred, Y_test)))
X_train_all = poly.transform(forecast)

pred_all = lr.predict(X_train_all)



plt.plot(forecast[:-15], y, color='red')

plt.plot(forecast, pred_all, linestyle='dashed')

plt.title('DEATHS of COVID-19')

plt.xlabel('Days since 1/22/2020')

plt.ylabel('Number of deaths')

plt.legend(['Death cases', 'Predictions']);
confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

confirmed
confirmed[confirmed['Country/Region'] == 'Brazil']
columns1 = confirmed.keys()

columns1
confirmed = confirmed.loc[:, columns1[4]:columns1[-1]]

confirmed
confirmed['1/22/20'].sum(), confirmed['8/12/20'].sum()
dates1 = confirmed.keys()

y_c = []

for i in dates1:

    y_c.append(confirmed[i].sum())
print(y_c)
y_c = np.array(y_c).reshape(-1,1)

y_c
x_c = np.arange(len(dates1)).reshape(-1,1)

x_c
forecast_c = np.arange(len(dates1) + 15).reshape(-1,1)

forecast_c
x_c.shape, y_c.shape, forecast_c.shape 
X_train_c, X_test_c, Y_train_c, Y_test_c = train_test_split(x_c, y_c, test_size = 0.10, shuffle = False)
poly_c = PolynomialFeatures(degree = 4)

X_train_poly_c = poly_c.fit_transform(X_train_c)

X_test_poly_c = poly_c.transform(X_test_c)
X_train_c.shape, X_test_c.shape, X_train_poly_c.shape, X_test_poly_c.shape
lr_c = LinearRegression()

lr_c.fit(X_train_poly_c, Y_train_c)
poly_pred_c = lr_c.predict(X_test_poly_c)

plt.plot(poly_pred_c, linestyle = 'dashed')

plt.plot(Y_test_c)
print('MAE:', mean_absolute_error(poly_pred_c, Y_test_c))

print('MSE:', mean_squared_error(poly_pred_c, Y_test_c))

print('RMSE:', np.sqrt(mean_absolute_error(poly_pred_c, Y_test_c)))
X_train_all_c = poly_c.transform(forecast_c)

pred_all_c = lr_c.predict(X_train_all_c)



plt.plot(forecast_c[:-15], y_c, color='red')

plt.plot(forecast_c, pred_all_c, linestyle='dashed')

plt.title('CONFIRMED of COVID-19')

plt.xlabel('Days since 1/22/2020')

plt.ylabel('Number of confirmed')

plt.legend(['Confirmed cases', 'Predictions']);
recovered = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

recovered
recovered[recovered['Country/Region'] == 'Brazil']
columns2 = recovered.keys()

columns2
recovered = recovered.loc[:, columns2[4]:columns2[-1]]

recovered
recovered['1/22/20'].sum(), recovered['8/12/20'].sum()
dates2 = recovered.keys()

y_r = []

for i in dates2:

    y_r.append(recovered[i].sum())
print(y_r)
y_r = np.array(y_r).reshape(-1,1)

y_r
x_r = np.arange(len(dates2)).reshape(-1,1)

x_r
forecast_r = np.arange(len(dates2) + 15).reshape(-1,1)

forecast_r
x_r.shape, y_r.shape, forecast_r.shape
X_train_r, X_test_r, Y_train_r, Y_test_r = train_test_split(x_r, y_r, test_size = 0.30, shuffle = False)
poly_r = PolynomialFeatures(degree = 3)

X_train_poly_r = poly_r.fit_transform(X_train_r)

X_test_poly_r = poly_r.transform(X_test_r)
X_train_r.shape, X_test_r.shape, X_train_poly_r.shape, X_test_poly_r.shape
lr_r = LinearRegression()

lr_r.fit(X_train_poly_r, Y_train_r)
poly_pred_r = lr_r.predict(X_test_poly_r)

plt.plot(poly_pred_r, linestyle = 'dashed')

plt.plot(Y_test_r)
print('MAE:', mean_absolute_error(poly_pred_r, Y_test_r))

print('MSE:', mean_squared_error(poly_pred_r, Y_test_r))

print('RMSE:', np.sqrt(mean_absolute_error(poly_pred_r, Y_test_r)))
X_train_all_r = poly_r.transform(forecast_r)

pred_all_r = lr_r.predict(X_train_all_r)



plt.plot(forecast_r[:-15], y_r, color='red')

plt.plot(forecast_r, pred_all_r, linestyle='dashed')

plt.title('RECOVERED of COVID-19')

plt.xlabel('Days since 1/22/2020')

plt.ylabel('Number of recovered')

plt.legend(['Recovered', 'Predictions']);