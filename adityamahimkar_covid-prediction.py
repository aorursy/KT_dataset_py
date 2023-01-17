import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
confirmed = pd.read_csv('../input/covid-dataset/time_series_covid19_confirmed_global.csv')
deaths = pd.read_csv('../input/covid-dataset/time_series_covid19_deaths_global.csv')
recovered = pd.read_csv('../input/covid-dataset/time_series_covid19_recovered_global.csv')
confirmed.head()
world_confirmed = confirmed.drop(labels=['Province/State', 'Lat', 'Long', 'Country/Region'], axis=1)
world_confirmed.head()
world_confirmed_T = world_confirmed.T
world_confirmed_T['Total'] = world_confirmed_T[0]
for col in world_confirmed_T.columns:
    if col == 0:
        continue
    world_confirmed_T['Total'] += world_confirmed_T[col]
world_confirmed_T.reset_index(inplace=True)
world_confirmed_T
world_confirmed = world_confirmed_T[['index', 'Total']]
world_confirmed = world_confirmed.rename(columns={'index':'Date'})
world_confirmed
train_c = world_confirmed[:140]
test_c = world_confirmed[140:]
train_time = train_c.Date
train_time = pd.to_datetime(train_time)
train_c = train_c.drop(labels=['Date'], axis=1)
train_c.index = train_time
test_time = test_c.Date
test_time = pd.to_datetime(test_time)
test_c = test_c.drop(labels=['Date'], axis=1)
test_c.index = test_time
train_c.head()
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(train_c['Total'], trend='mul', seasonal_periods=12).fit()
pred = model.forecast(len(test_c))
plt.figure(figsize=(10, 6))
plt.plot(train_c.index, train_c.Total, label='train')
plt.plot(test_c.index, test_c.Total, label='test')
plt.plot(test_c.index, pred, label='predict')
deaths.head()
world_deaths = deaths.drop(labels=['Province/State', 'Lat', 'Long', 'Country/Region'], axis=1)
world_deaths.head()
world_deaths_T = world_deaths.T
world_deaths_T['Total'] = world_deaths_T[0]
for col in world_deaths_T.columns:
    if col == 0:
        continue
    world_deaths_T['Total'] += world_deaths_T[col]
world_deaths_T.reset_index(inplace=True)
world_deaths_T
world_deaths = world_deaths_T[['index', 'Total']]
world_deaths = world_deaths.rename(columns={'index':'Date'})
world_deaths
train_d = world_deaths[0:140]
test_d = world_deaths[140:]
train_time = train_d.Date
train_time = pd.to_datetime(train_time)
train_d = train_d.drop(labels=['Date'], axis=1)
train_d.index = train_time
test_time = test_d.Date
test_time = pd.to_datetime(test_time)
test_d = test_d.drop(labels=['Date'], axis=1)
test_d.index = test_time
train_d.head()
x = np.arange(len(train_d))
x = x.reshape(-1, 1)
y_train = np.array(train_d.Total)
y_train
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_train = poly.fit_transform(x)
x1 = np.arange(len(train_d)+1, len(train_d)+len(test_d)+1)
x1 = x1.reshape(-1, 1)
X_test = poly.transform(x1)
y_test = np.array(test_d.Total)
y_test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
model = LinearRegression()
model.fit(X_train, y_train)
y_base_predict = model.predict(X_train)
plt.scatter(train_d.index, y_train, label='given data')
plt.plot(train_d.index, y_base_predict, color='r', label='line of best fit')
plt.legend()
plt.show()
y_pred = model.predict(X_test)
plt.plot(train_d.index, y_train, label='Train')
plt.plot(test_d.index, y_test, label='Test')
plt.plot(test_d.index, y_pred, label='Predict')
plt.legend()
plt.show()
recovered.head()
world_recovered = recovered.drop(labels=['Province/State', 'Lat', 'Long', 'Country/Region'], axis=1)
world_recovered.head()
world_recovered_T = world_recovered.T
world_recovered_T['Total'] = world_recovered_T[0]
for col in world_recovered_T.columns:
    if col == 0:
        continue
    world_recovered_T['Total'] += world_recovered_T[col]
world_recovered_T.reset_index(inplace=True)
world_recovered_T
world_recovered = world_recovered_T[['index', 'Total']]
world_recovered = world_recovered.rename(columns={'index':'Date'})
world_recovered
train_r = world_recovered[0:140]
test_r = world_recovered[140:]
train_time = train_r.Date
train_time = pd.to_datetime(train_time)
train_r = train_r.drop(labels=['Date'], axis=1)
train_r.index = train_time
test_time = test_r.Date
test_time = pd.to_datetime(test_time)
test_r = test_r.drop(labels=['Date'], axis=1)
test_r.index = test_time
train_r.head()
model = ExponentialSmoothing(train_r['Total'], trend='mul', seasonal='mul').fit()
pred = model.forecast(len(test_r))
plt.figure(figsize=(10, 6))
plt.plot(train_r.index, train_r.Total, label='train')
plt.plot(test_r.index, test_r.Total, label='test')
plt.plot(test_r.index, pred, label='predict')
