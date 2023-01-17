import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker 
from datetime import datetime, timedelta,date
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline


df_stats = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')

df_stats.head()
df_stats = df_stats.loc[df_stats['iso_code'] == 'OWID_WRL']
df_stats_cases = df_stats[['date','new_cases']].fillna(0)
df_stats_cases = df_stats_cases.groupby(['date']).sum()
df_stats_cases.tail(100)

f = plt.figure(figsize=(15,10))
# Grid
plt.grid(lw = 1, ls = ':', c = "0.4", which = 'major')


marker_style = dict(linewidth=3, linestyle='-', marker='o',markersize=5, markerfacecolor='#ffffff')
plt.plot(df_stats_cases,color="green",**marker_style, label="Actual Curve")
# plt.plot(e_history, color='red')
# plt.show()
plt.show()

df_stats_cases_diff = df_stats_cases - df_stats_cases.shift(12)
df_stats_cases_diff.dropna(inplace=True)

f = plt.figure(figsize=(15,10))
# Grid
plt.grid(lw = 1, ls = ':', c = "0.4", which = 'major')

df_stats_cases_diff.index = pd.to_datetime(df_stats_cases_diff.index)

marker_style = dict(linewidth=3, linestyle='-', marker='o',markersize=5, markerfacecolor='#ffffff')
plt.plot(df_stats_cases_diff,**marker_style, label="Actual Curve")

plt.show()

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(df_stats_cases_diff, nlags=20)
lag_pacf = pacf(df_stats_cases_diff, nlags=20, method='ols')
marker_style = dict(linewidth=3, linestyle='-', marker='o',markersize=5, markerfacecolor='#ffffff')
plt.plot(df_stats_cases,color="green",**marker_style, label="Actual Curve")
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf,**marker_style,)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_stats_cases_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_stats_cases_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation')

plt.subplot(122)
plt.plot(lag_pacf,**marker_style,)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.9/np.sqrt(len(df_stats_cases_diff)),linestyle='--',color='gray')
plt.axhline(y=1.9/np.sqrt(len(df_stats_cases_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation')
plt.tight_layout()
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_acf(df_stats_cases_diff, lags=40, ax=ax1)
fig = sm.graphics.tsa.plot_pacf(df_stats_cases_diff, lags=40, ax=ax2)


model = ARIMA(df_stats_cases, order=(4, 1, 2))  
results_ARIMA = model.fit(disp=-1)
plt.plot(df_stats_cases_diff)
results_ARIMA.fittedvalues.index = pd.to_datetime(results_ARIMA.fittedvalues.index)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.show()
# predicting the fitted values

fitted_diff = results_ARIMA.fittedvalues - results_ARIMA.fittedvalues.shift()
fitted_diff.dropna(inplace=True)
plt.plot(fitted_diff)
plt.show()

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(fitted_diff, nlags=20)
lag_pacf = pacf(fitted_diff, nlags=20, method='ols')

marker_style = dict(linewidth=3, linestyle='-', marker='o',markersize=5, markerfacecolor='#ffffff')
plt.plot(df_stats_cases,color="green",**marker_style, label="Actual Curve")
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf,**marker_style,)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(fitted_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(fitted_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf,**marker_style,)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.9/np.sqrt(len(fitted_diff)),linestyle='--',color='gray')
plt.axhline(y=1.9/np.sqrt(len(fitted_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

fitted_model = ARIMA(results_ARIMA.fittedvalues, order=(2, 1, 2))  
results_fitted = fitted_model.fit(disp=-1)
plt.plot(fitted_diff)
plt.plot(results_fitted.fittedvalues, color='red')
plt.show
# goto SECTION
results_ARIMA.forecast(steps=100)[0]
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()
predictions_ARIMA_diff.tail()
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())
predictions_ARIMA_log = pd.Series(df_stats_cases['new_cases'].iloc[0], index=df_stats_cases.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

plt.plot(df_stats_cases)
plt.plot(predictions_ARIMA_log, color='r')
plt.show()

# f = plt.figure(figsize=(15,10))
# ax = f.add_subplot(111)
X = df_stats_cases.values

predictions = list()
# model = ARIMA(X, order=(6,1,5))
model_fit = model.fit(disp=0)

forecast = results_ARIMA.forecast(steps=30)[0]

history = [x for x in X]
day = 1
for yhat in forecast:
	print('+%d days: %f' % (day, yhat))
	history.append(yhat)
	day += 1
 
model_fit.summary()
   
# plot_data = temp_data.copy()
# plot_data.index = pd.to_datetime(plot_data.index)

start_date = df_stats_cases.index[0]
dates = pd.date_range(start_date, periods=len(df_stats_cases)+30, freq='D')
df_dates = pd.DataFrame(index=dates)
history_log = pd.Series(history, index=df_dates.index)

df_stats_cases.index = pd.to_datetime(df_stats_cases.index)

f = plt.figure(figsize=(15,10))
# Grid
plt.grid(lw = 1, ls = ':', c = "0.4", which = 'major')

marker_style = dict(linewidth=3, linestyle='-', marker='o',markersize=5, markerfacecolor='#ffffff')
plt.plot(df_stats_cases,color="green",**marker_style, label="Actual Curve")
plt.plot(history_log, color='red')
plt.show()
df_stats = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
exog = df_stats[['date', 'new_cases', 'new_tests_smoothed']]
exog = exog.groupby(['date']).sum()
exog = exog.replace(0, np.nan).ffill(axis=1).ffill(axis=0).iloc[:,1:]
# exog.iloc[-1,0] = exog.iloc[-2,0]


f = plt.figure(figsize=(15,10))
# Grid
plt.grid(lw = 1, ls = ':', c = "0.4", which = 'major')

exog.index = pd.to_datetime(exog.index)

plt.plot(exog)

# EXOG PREDICTIONS #

exog2 = exog

exog_diff = exog2 - exog2.shift()
exog_diff.dropna(inplace=True)
plt.plot(exog_diff)
plt.show()

#ACF and PACF plots:
lag_acf = acf(exog_diff, nlags=20)
lag_pacf = pacf(exog_diff, nlags=20, method='ols')

marker_style = dict(linewidth=3, linestyle='-', marker='o',markersize=5, markerfacecolor='#ffffff')
plt.plot(exog,color="green",**marker_style, label="Actual Curve")
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf,**marker_style,)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(exog_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(exog_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf,**marker_style,)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.9/np.sqrt(len(exog_diff)),linestyle='--',color='gray')
plt.axhline(y=1.9/np.sqrt(len(exog_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

e_model = ARIMA(exog2, order=(1, 0, 1))  
e_model_fit = e_model.fit(disp=-1)
plt.plot(exog_diff)
plt.plot(exog2, color='red')
plt.show()
e_forecast = e_model_fit.forecast(steps=100)[0]

e_X = exog.values

e_history = [x for x in X]
day = 1
for yhat in e_forecast:
	print('Day %d: %f' % (day, yhat))
	e_history.append(yhat)
	day += 1
    
exog.tail()

tests_pred = pd.Series(e_history, copy=True).tail(100).values
tests_pred = np.array(tests_pred, dtype=float)
start_date = exog.index[-1]
dates = pd.date_range(start_date, periods=100, freq='D')
df_dates = pd.DataFrame(index=dates)
exog_log = pd.Series(tests_pred, index=df_dates.index)

exog_log
# ####################################
f = plt.figure(figsize=(15,10))
# Grid
plt.grid(lw = 1, ls = ':', c = "0.4", which = 'major')


marker_style = dict(linewidth=3, linestyle='-', marker='o',markersize=5, markerfacecolor='#ffffff')
plt.plot(exog,color="green",**marker_style, label="Actual Curve")
plt.plot(exog_log, color='red')
plt.show()
# SARIMAX

s_model = sm.tsa.statespace.SARIMAX(
    endog = df_stats_cases.iloc[:,0],
    order = (4,1,2),
    exog = exog,
    seasonal_order=(2, 1, 1, 14),
    enforce_stationarity=True,
    enforce_invertibility=False,
    dynamic=True
)

s_model_fit = s_model.fit()

s_pred = s_model_fit.get_prediction(
    start = start_date,
    end = start_date+ timedelta(days=100),
    exog=exog_log
)

s_pred_ci = s_pred.conf_int()

print(s_pred_ci)

df_stats_cases.index = pd.to_datetime(df_stats_cases.index)

f = plt.figure(figsize=(15,10))
marker_style = dict(linewidth=3, linestyle='-', marker='o',markersize=5, markerfacecolor='#ffffff')

# Graph
fig, ax = plt.subplots(figsize=(15,10))
ax.grid(lw = 1, ls = ':', c = "0.4", which = 'major')
ax.set(title='New Cases prediction, exog=tests', xlabel='Date', ylabel='Cases')

# Plot data points
df_stats_cases.plot(ax=ax,color="green", **marker_style, label='Actual Curve')

# Plot predictions
s_pred.predicted_mean.plot(ax=ax, style='r--', label='forecast')
ax.fill_between(s_pred_ci.index, s_pred_ci.iloc[:,0], s_pred_ci.iloc[:,1], color='r', alpha=0.1)
# predict_dy.predicted_mean.loc['1977-07-01':].plot(ax=ax, style='g', label='Dynamic forecast (1978)')
# ci = predict_dy_ci.loc['1977-07-01':]
# ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)

legend = ax.legend(loc='lower right')
log_df = pd.DataFrame(exog_log)
log_df.columns=['new_tests_smoothed']
# log_df
full_tests = pd.concat([exog,log_df])

log_df_cases = pd.DataFrame(exog_log)
log_df_cases.columns=['new_cases']

full_cases = pd.concat([df_stats_cases,log_df_cases])


full_log = pd.concat([full_cases,full_tests],axis=1)
full_log['infection_rate'] = full_log.apply(lambda row: row['new_cases']/row['new_tests_smoothed'] if row['new_cases']/row['new_tests_smoothed'] <= 1 else 0.9, axis = 1) 
exog_log_2 = full_log.iloc[:,1:].tail(100)

prev_full_log = pd.concat([df_stats_cases,exog],axis=1)
prev_full_log['infection_rate'] = prev_full_log.apply(lambda row: row['new_cases']/row['new_tests_smoothed'] if row['new_cases']/row['new_tests_smoothed'] <= 1 else 0.9, axis = 1) 
prev_exog_log_2 = prev_full_log.iloc[:,1:]

prev_exog_log_2
# exog_log_2

s_model = sm.tsa.statespace.SARIMAX(
    endog = df_stats_cases.iloc[:,0],
    order = (4,1,2),
    exog = prev_exog_log_2,
    seasonal_order=(0, 1, 1, 14),
    enforce_stationarity=True,
    enforce_invertibility=False,
    dynamic=True
)

s_model_fit = s_model.fit()

s_pred = s_model_fit.get_prediction(
    start = start_date,
    end = start_date+ timedelta(days=100),
    exog=exog_log_2
)

s_pred_ci = s_pred.conf_int(alpha=0.1)

print(s_pred_ci)

s_model_fit.summary()
s_pred_ci = s_pred.conf_int(alpha=0.1)
df_stats_cases.index = pd.to_datetime(df_stats_cases.index)

f = plt.figure(figsize=(15,10))
marker_style = dict(linewidth=3, linestyle='-', marker='o',markersize=5, markerfacecolor='#ffffff')

# Graph
fig, ax = plt.subplots(figsize=(15,10))
ax.grid(lw = 1, ls = ':', c = "0.4", which = 'major')
ax.set(title='New Cases prediction, exog=tests,infection_rate', xlabel='Date', ylabel='Cases')

# Plot data points
df_stats_cases.plot(ax=ax,color="green", **marker_style, label='Actual Curve')

# Plot predictions
s_pred.predicted_mean.plot(ax=ax, style='r--', label='forecast')
ax.fill_between(s_pred_ci.index, s_pred_ci.iloc[:,0], s_pred_ci.iloc[:,1], color='r', alpha=0.1)
# predict_dy.predicted_mean.loc['1977-07-01':].plot(ax=ax, style='g', label='Dynamic forecast (1978)')
# ci = predict_dy_ci.loc['1977-07-01':]
# ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)

legend = ax.legend(loc='lower right')











