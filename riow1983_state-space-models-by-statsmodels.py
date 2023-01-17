import numpy as np

import pandas as pd

from scipy import stats



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.rcParams['figure.figsize'] = 15, 6



import statsmodels.api as sm
# reference: https://logics-of-blue.com/wp-content/uploads/2017/05/python-state-space-models.html
df = pd.read_csv("../input/rossmann-store-sales/train.csv", parse_dates = True, index_col = 'Date')

df.head()
data = df['Sales'].groupby('Date').sum()

data.head()
thres = '2014-12-31'

train = data[data.index <= thres]

test = data[data.index > thres]
train.shape
test.shape
test.tail()
plt.plot(train)
plt.plot(test)
def mape(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true))
def rmspe(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.sqrt(np.mean(((y_true - y_pred) / y_true)**2))
mod_local_level = sm.tsa.UnobservedComponents(train, 'local level', freq='D')



res_local_level = mod_local_level.fit()



print(res_local_level.summary())

plt.rcParams['figure.figsize'] = 24, 20

res_local_level.plot_components();
pred_local_level = res_local_level.predict('2015-01-01', '2015-07-31')

plt.plot(test)

plt.plot(pred_local_level, 'r');

print("MAPE = ", mape(test, pred_local_level))

print("RMSPE = ", rmspe(test, pred_local_level))
mod_trend = sm.tsa.UnobservedComponents(train, 'local linear trend', freq='D')



res_trend = mod_trend.fit()



print(res_trend.summary())

plt.rcParams['figure.figsize'] = 24, 20

res_trend.plot_components();
pred_trend = res_trend.predict('2015-01-01', '2015-07-31')

plt.rcParams['figure.figsize'] = 24, 6

plt.plot(test)

plt.plot(pred_trend, 'r');

print("MAPE = ", mape(test, pred_trend))

print("RMSPE = ", rmspe(test, pred_trend))
mod_season_local_level = sm.tsa.UnobservedComponents(train, 'local level', freq='D', seasonal=12)



res_season_local_level = mod_season_local_level.fit()



print(res_season_local_level.summary())

plt.rcParams['figure.figsize'] = 24, 20

res_season_local_level.plot_components();
pred_season_local_level = res_season_local_level.predict('2015-01-01', '2015-07-31')

plt.rcParams['figure.figsize'] = 24, 6

plt.plot(test)

plt.plot(pred_season_local_level, 'r');

print("MAPE = ", mape(test, pred_season_local_level))

print("RMSPE = ", rmspe(test, pred_season_local_level))
mod_season_trend = sm.tsa.UnobservedComponents(train, 'local linear trend', freq='D', seasonal=12)



res_season_trend = mod_season_trend.fit()



print(res_season_trend.summary())

plt.rcParams['figure.figsize'] = 24, 20

res_season_trend.plot_components();
pred_season_trend = res_season_trend.predict('2015-01-01', '2015-07-31')

plt.rcParams['figure.figsize'] = 24, 6

plt.plot(test)

plt.plot(pred_season_trend, 'r');

print("MAPE = ", mape(test, pred_season_trend))

print("RMSPE = ", rmspe(test, pred_season_trend))
mod_season_trend = sm.tsa.UnobservedComponents(train, 'local linear trend', freq='D', seasonal=12)



#res_season_trend = mod_season_trend.fit()

res_season_trend = mod_season_trend.fit(

    method='bfgs',

    maxiter=500,

    start_params=mod_season_trend.fit(method='nm', maxiter=500).params

)



print(res_season_trend.summary())

plt.rcParams['figure.figsize'] = 24, 20

res_season_trend.plot_components();
pred_season_trend = res_season_trend.predict('2015-01-01', '2015-07-31')

plt.rcParams['figure.figsize'] = 24, 6

plt.plot(test)

plt.plot(pred_season_trend, 'r');

print("MAPE = ", mape(test, pred_season_trend))

print("RMSPE = ", rmspe(test, pred_season_trend))