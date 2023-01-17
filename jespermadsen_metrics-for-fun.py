import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

# Pæne dataserier

x = np.arange(0,20,1)

actual = np.random.randint(20, size=20) # baseline y

forecast = np.random.randint(20, size=20) # eksempel på yhat

low_error = actual + np.random.randint(-5,5, size=20) # eksempel på lav forskel

high_error = actual + np.random.randint(-30,30, size=20) # eksempel på høj forskel
fig,((ax,a2),(a3,a4)) = plt.subplots(2,2, figsize=(20,10))

ax.plot(actual, c='g')

ax.grid()

ax.legend(['actual'])

ax.set_title('baseline')

ax.set_ylim(-20,35)



a2.fill_between(x,actual, forecast, alpha=0.2)

a2.plot(actual, c='g')

a2.plot(forecast, c='y')

a2.grid()

a2.legend(['actual','forecast'])

a2.set_title('forecast')

a2.set_ylim(-20,35)



a3.fill_between(x,actual, low_error, alpha=0.2)

a3.plot(actual, c='g')

a3.plot(low_error, c='y')

a3.grid()

a3.legend(['actual','low_error'])

a3.set_title('low_error')

a3.set_ylim(-20,35)



a4.fill_between(x,actual, high_error, alpha=0.2)

a4.plot(actual, c='g')

a4.plot(high_error, c='y')

a4.grid()

a4.legend(['actual','high_error'])

a4.set_title('high_error')

a4.set_ylim(-20,35)



plt.show()
def _error(actual,forecast):

    return actual - forecast
error_forecast = _error(actual,forecast)

error_low_error = _error(actual, low_error)

error_high_error = _error(actual, high_error)


fig, ((ax,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(20,10))

ax2.plot(error_forecast)

ax2.hlines(0,0,20, colors='k')

ax2.grid()

ax2.set_title('forecast forskel')

ax2.set_ylim(-30,30)

ax2.set_xlim(0,20)



ax3.plot(error_low_error)

ax3.hlines(0,0,20, colors='k')

ax3.grid()

ax3.set_title('low_error forskel')

ax3.set_ylim(-30,30)

ax3.set_xlim(0,20)



ax4.plot(error_high_error)

ax4.hlines(0,0,20, colors='k')

ax4.grid()

ax4.set_title('high_error forskel')

ax4.set_ylim(-30,30)

ax4.set_xlim(0,20)

fig.delaxes(ax)

plt.show()
def _mean_error_score():

    print('Gennemsnit af forskellene')

    print('-'*40)

    print('forecast  : {}'.format(np.mean(forecast)))

    print('low error : {}'.format(np.mean(low_error)))

    print('high error: {}'.format(np.mean(high_error)))

    print('-'*50)
_mean_error_score()
def _abs_mean_error_score():

    print('Gennemsnit af absolutte forskelle')

    print('-'*40)

    print('forecast  : {}'.format(np.mean(np.abs(forecast))))

    print('low error : {}'.format(np.mean(np.abs(low_error))))

    print('high error: {}'.format(np.mean(np.abs(high_error))))

    print('-'*50)
_abs_mean_error_score()
def _squared_error(actual, forecast):

    return ((actual - forecast)**2)
se_forecast = _squared_error(actual, forecast)

se_low_error = _squared_error(actual, low_error)

se_high_error = _squared_error(actual, high_error)
fig, ((ax,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(20,10))

ax2.plot(se_forecast)

ax2.grid()

ax2.set_title('forecast error^2')

ax2.set_ylim(0,200)



ax3.plot(se_low_error)

ax3.grid()

ax3.set_title('low_error forskel^2')

ax3.set_ylim(0,200)



ax4.plot(se_high_error)

ax4.grid()

ax4.set_title('high_error forskel^2')

ax4.set_ylim(0,200)



fig.delaxes(ax)

plt.show()
def _mean_squared_error(actual, forecast):

    return np.mean((actual - forecast)**2)
MSE_forecast = _mean_squared_error(actual, forecast)

MSE_low_error = _mean_squared_error(actual, low_error)

MSE_high_error = _mean_squared_error(actual, high_error)
def _mse_scores():

    print('MSE Scores:')

    print('-'*40)

    print('forecast  : {}'.format(MSE_forecast))

    print('low_error : {}'.format(MSE_low_error))

    print('high_error: {}'.format(MSE_high_error))

    print('-'*50)
_mse_scores()
def _rmse(actual, forecast):

    return _mean_squared_error(actual,forecast)**.5
def _rmse_scores():

    RMSE_forecast = _rmse(actual, forecast)

    RMSE_low_error = _rmse(actual, low_error)

    RMSE_high_error = _rmse(actual, high_error)

    print('RMSE Scores:')

    print('-'*40)

    print('forecast  : {}'.format(round(RMSE_forecast,2)))

    print('low_error : {}'.format(round(RMSE_low_error,2)))

    print('high_error: {}'.format(round(RMSE_high_error,2)))

    print('-'*50)
_rmse_scores()
from datetime import datetime



datelist = pd.date_range(end = datetime.today(), periods = 40)

actualx2 = np.append(np.random.randint(20, size=20),actual)

actual_df = pd.DataFrame(list(zip(datelist, actualx2)), columns=['date','sales'])

forecast_df = pd.DataFrame(list(zip(datelist[20:], forecast)), columns=['date','sales_forecast'])

low_error_df = pd.DataFrame(list(zip(datelist[20:], low_error)), columns=['date','sales_low_error'])

high_error_df = pd.DataFrame(list(zip(datelist[20:], high_error)), columns=['date','sales_high_error'])
fig,((ax,a2),(a3,a4)) = plt.subplots(2,2, figsize=(20,10))

actual_df.plot('date','sales',c='g', ax=ax)

ax.grid()

ax.legend(['actual'])

ax.set_title('baseline')

ax.axhline(y=-5, xmin=0.05, xmax=0.95, lw=10)

ax.axhline(y=-2, xmin=0.5, xmax=0.95, c='y', lw=10)

ax.axhline(y=-8, xmin=0.05, xmax=0.49, lw=10, c='r')

ax.set_ylim(-20,35)



a2.fill_between(forecast_df['date'],actual_df['sales'][20:], forecast_df['sales_forecast'], alpha=0.2)

actual_df.plot('date','sales',c='g', ax=a2)

a2.plot(forecast_df['date'], forecast_df['sales_forecast'], c='y')

a2.grid()

a2.legend(['actual','forecast'])

a2.set_title('forecast')

a2.set_ylim(-20,35)



#a3.fill_between(x,actual, low_error, alpha=0.2)

a3.fill_between(forecast_df['date'],actual_df['sales'][20:], low_error_df['sales_low_error'], alpha=0.2)

actual_df.plot('date','sales',c='g', ax=a3)

a3.plot(low_error_df['date'], low_error_df['sales_low_error'], c='y')

a3.grid()

a3.legend(['actual','low_error'])

a3.set_title('low_error')

a3.set_ylim(-20,35)



#a4.fill_between(x,actual, high_error, alpha=0.2)

a4.fill_between(forecast_df['date'],actual_df['sales'][20:], high_error_df['sales_high_error'], alpha=0.2)

actual_df.plot('date','sales',c='g', ax=a4)

a4.plot(high_error_df['date'], high_error_df['sales_high_error'], c='y')

a4.grid()

a4.legend(['actual','high_error'])

a4.set_title('high_error')

a4.set_ylim(-20,35)



plt.show()
def _RMSSE(actual, forecast):

    numerator = _mean_squared_error(actual['sales'][len(forecast):].values,forecast)

    #print('numerator: {}'.format(numerator))

    denominator = np.mean((actual[:20] - actual[:20].shift(1)).dropna()['sales']**2)

    #print('denominator: {}'.format(denominator))

    return (numerator/denominator)**0.5
def _rmsse_scores():

    rmsse_forecast = _RMSSE(actual_df, forecast)

    rmsse_low_error = _RMSSE(actual_df, low_error)

    rmsse_high_error = _RMSSE(actual_df, high_error)

    print('RMSSE Scores:')

    print('-'*40)

    print('forecast  : {}'.format(round(rmsse_forecast,2)))

    print('low_error : {}'.format(round(rmsse_low_error,2)))

    print('high_error: {}'.format(round(rmsse_high_error,2)))

    print('-'*50)
_rmsse_scores()
_mean_error_score()

_abs_mean_error_score()

_mse_scores()

_rmse_scores()

_rmsse_scores()