%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn import *

from datetime import datetime

import calendar

# 

air = pd.read_csv('../input/air_visit_data.csv', parse_dates=[1])

air.set_index(['visit_date'], inplace=True)

air.index.name=None

air.drop('air_store_id',axis=1,inplace=True)

df2=pd.DataFrame()

df2['visit_total'] = air.groupby(air.index,squeeze=True,sort=True)['visitors'].sum()

df2['visit_mean'] = air.groupby(air.index,squeeze=True,sort=True)['visitors'].mean()

df2['reserv_cnt'] = air.groupby(air.index,squeeze=True,sort=True)['visitors'].count()

air=df2;del df2



#Get the date info with dow and holidays

hol=pd.read_csv('../input/date_info.csv', parse_dates=True).rename(columns={'calendar_date':'visit_date'})

hol['visit_date'] = pd.to_datetime(hol['visit_date'])

hol.set_index(['visit_date'], inplace=True)

hol.index.name=None

hol.day_of_week = hol.day_of_week.apply(list(calendar.day_name).index)



#Get the test submission

test = pd.read_csv('../input/sample_submission.csv')

test['store_id'], test['visit_date'] = test['id'].str[:20], test['id'].str[21:]

test.set_index('visit_date', drop=True, inplace=True)

test.index.name=None
#Plot the cumulative visits

air['visit_total'].plot(legend=True);

air['reserv_cnt'].plot(legend=True, figsize=(15,4), secondary_y=True,

                      title='Visitors total and reservation count (with holidays)');

for x in hol.query('holiday_flg==1').index:

    _ = plt.axvline(x=x, color='k', alpha = 0.3);
air['visit_mean'].plot(figsize=(15,4), legend=True, title='Visitors mean (with holidays)')

air['reserv_cnt'].plot(legend=True, figsize=(15,4), secondary_y=True, title='Visitors total and reservation count (with holidays)');

for x in hol.query('holiday_flg==1').index:

    _ = plt.axvline(x=x, color='k', alpha = 0.3);
import statsmodels.api as sm  

from statsmodels.tsa.stattools import acf  

from statsmodels.tsa.stattools import pacf

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(air.visit_mean, freq=12)  

fig = plt.figure()  

fig = decomposition.plot()  

fig.set_size_inches(15, 8)
df2=air.join(hol)

df2[df2.holiday_flg==0].groupby(hol.day_of_week,squeeze=True,sort=True)['visit_mean'].sum()

#df2.day_of_week=df3.day_of_week.apply(lambda x: list(calendar.day_name)[x]) # equiv to air.sum(0)
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    

    #Determing rolling statistics

    rolmean = timeseries.rolling(window=12,center=False).mean();

    rolstd = timeseries.rolling(window=12,center=False).std();



    #Plot rolling statistics:

    fig = plt.figure(figsize=(15, 5))

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show()

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)



test_stationarity(air.visit_mean); #-3.796104

# Log is a minor improvement, meaning that the variance is stable

air.visit_mean_log= air.visit_mean.apply(lambda x: np.log(x))  

'''test_stationarity(air.visit_mean_log) #-3.830754'''

# Although I see no real global trend, 1st difference strongly improves stationarity

air['visit_mean_diff'] = air.visit_mean - air.visit_mean.shift(1)  

test_stationarity(air.visit_mean_diff.dropna(inplace=False)) #-6.608968e+00

# Seasonal difference: take a weekly season improves stationarity even more

air['visit_mean_seasonal'] = air.visit_mean - air.visit_mean.shift(7)

test_stationarity(air.visit_mean_seasonal.dropna(inplace=False)) #-7.196314e+00

# Seasonal and 1st difference is even better, but we were already well within the 1% confidence interval

air['visit_mean_seasonal_diff'] = air.visit_mean_diff - air.visit_mean_diff.shift(7)

test_stationarity(air.visit_mean_seasonal_diff.dropna(inplace=False)) #-9.427797e+00
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(air.visit_mean, lags=40, alpha=.05, ax=ax1)

ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_pacf(air.visit_mean, lags=40, alpha=.05, ax=ax2)

print("ACF and PACF of the visit mean:")
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(air.visit_mean_diff[1:], lags=40, alpha=.05, ax=ax1)
print("ACF and PACF of the 7-day differenced visit mean:")

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(air.visit_mean_seasonal[8:], lags=40, alpha=.05, ax=ax1)

ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_pacf(air.visit_mean_seasonal[8:], lags=40, alpha=.05, ax=ax2)
#sklearn.metrics .mean_squared_log_error seems to exist but I cannot load it..

from sklearn.metrics import mean_squared_error

def mean_squared_log_error(y_pred, y_true, **dict):

    '''Assume y_true starts earlier than y_pred, y_true is NaN free, and NaN in y_pred are only in the beginning'''

    indafterNaN = y_pred.first_valid_index()

    if (y_true.index[0] > y_pred.index[0]): return "Check indices of prediction and true value"

    ind1stcommon = y_true.index[y_true.index==y_pred.index[0]]

    indstart = max(indafterNaN, ind1stcommon)

    indend = y_true.index[-1]

    return mean_squared_error(np.log(y_true[indstart:indend]+1), 

                              np.log(y_pred[indstart:indend]+1) )**0.5



def plotSARIMAX(labels, pred):

    fig = plt.figure(figsize=(12, 8))

    layout = (2, 2)

    ax1 = plt.subplot2grid(layout, (0, 0), colspan=2)

    ax3 = plt.subplot2grid(layout, (1, 0))

    ax4 = plt.subplot2grid(layout, (1, 1))

    labels.plot(ax=ax1);

    pred.plot(ax=ax1, title='MSE: %.4f'% mean_squared_log_error(pred, labels))

    ax3 = sm.graphics.tsa.plot_acf(results.resid, lags=40, alpha=.05, ax=ax3, title="ACF of residuals")

    ax4 = sm.graphics.tsa.plot_pacf(results.resid, lags=40, alpha=.05, ax=ax4, title="PACF of residuals")

    plt.tight_layout()

    print("ACF and PACF of residuals")
from scipy.optimize import brute

from sklearn.metrics import mean_squared_error



def autoSARIMAX(endog, exog=None, date_train_end=None, pred_days=[-12,12], verbose=True,\

        ranges=(slice(1,3),slice(0,1),slice(1,3),  slice(0,2),slice(1,2),slice(1,2),slice(7,8))):

    #Instantiate my version of the grid with parameters and scores

    global grid

    grid = []

    #Get indices up to which you do train and prediction 

    if date_train_end is None:

        ind_train = endog.index[-1]

    else:

        ind_train = np.where(endog.index==date_train_end)[0][0]

    #Brute optimization

    resultsbrute = brute(runSARIMAX, ranges=ranges, args=(endog,exog,(ind_train,pred_days),), full_output=True, finish=None)

    #First coefficients run two times for some reason or another

    del grid[0]

    #Print/Plot results

    if verbose:

        print("Best parameters: {}".format([int(p) for p in resultsbrute[0]]))

        print("Best score:          {}".format(resultsbrute[1]))

        gr = plotautoSARIMAX(resultsbrute, verbose)

    return resultsbrute, gr



def plotautoSARIMAX(resultsbrute, verbose=True):

    #Print/Plot results

    if not verbose: return None

    #Plot scores by parameter values

    gr = pd.DataFrame({'params':[''.join(str(n) for n in g[0]) for g in grid], 'score': [row[1] for row in grid], 'aic': [row[2] for row in grid]})

    print("All parameters and scores: \n")

    print(gr.head(1000).to_string())

    ax1 = gr.plot('params','score',rot=90, grid=True, figsize=(15,4))

    ax2 = gr.plot('params','aic',rot=90, secondary_y=True,ax=ax1)

    ax1.set_ylabel('Score');ax2.set_ylabel('AIC');

    plt.xticks(range(len(gr)), gr.params, rotation=90);

    return gr



def runSARIMAX(coeffs, *args):

    endog = args[0]

    exog = args[1]

    #Process the row indices for training and prediction

    ind_train = args[2][0]

    pred_days = args[2][1]

    ind_pred = [len(endog)+pred_days[0], len(endog)+pred_days[1]]

    if ind_pred[0] > ind_train: 

        #ind_pred[0]=ind_train

        raise ValueError('Make sure prediction bounds begin at least at len(endog): pred_days[0] must be <= %i ' % (ind_train-len(endog)))

    exog_train, exog_pred, start_params = None, None, list()

    if exog is not None:

        if ind_pred[1] > len(exog):

            raise ValueError('Make sure prediction bounds end  <= len(exog): pred_days[1] must be <= %i ' % (len(exog)-len(endog)))

        exog_train = exog[:ind_train]

        exog_cols = 1 if len(exog.shape) == 1 else exog.shape[1]

        start_params.extend(0.1*np.ones(exog_cols-1))

        exog_pred = exog[ind_pred[0]-1:ind_pred[1]]

        exog_pred = pd.DataFrame(exog_pred)

        

    #Get the hyperparameters

    order = coeffs[0:3].tolist()

    seasonal_order = coeffs[3:7].tolist()

    trend = 'c' if (order[1]==0) else 'n'

    #Train SARIMAX and fit it on data, predict to get scores

    try:        

        mod = sm.tsa.statespace.SARIMAX(endog[:ind_train], exog_train, \

                                        trend=trend, order=order, seasonal_order=seasonal_order)

        start_params.extend(0.1*np.ones( len(mod.params_complete)))

        fit = mod.fit(start_params=start_params)

        pred = fit.predict(start=ind_pred[0], end=ind_pred[1], exog=exog_pred)

        aic = fit.aic

        score = mean_squared_log_error(pred[:-pred_days[0]], endog[ind_pred[0]:])        

        if np.isnan(aic): aic, score = np.inf, np.inf

    except:  #Tip: Try to set starting paramenters in .fit()

        import sys        

        print("Error:", sys.exc_info())        

        print("{},{},'{}', len(start_params)={}\n".format(coeffs[0:3], coeffs[3:], trend, len(start_params)))

        aic, score = np.inf, np.inf

    #Sorry but I don't like the grid in the output of brute resultsbrute[2]

    global grid

    grid.append([coeffs,score,aic])

    return score



#Quick example

#resbrute, gr = autoSARIMAX(endog=air.visit_mean, exog=None, date_train_end="2017-03-26", pred_days=[-28,66],\

#                             ranges=(slice(1,2),slice(0,1),slice(1,2),  slice(0,2),slice(1,2),slice(1,2),slice(7,8)))



#resbrute, gr=autoSARIMAX(endog=air.visit_mean, exog=hol.holiday_flg, date_train_end="2017-03-26", pred_days = [-28,39],\

#                    ranges=(slice(1,2),slice(0,1),slice(1,2),  slice(0,1),slice(1,2),slice(1,2),slice(7,8)))
resbrute, gr = autoSARIMAX(endog=air.visit_mean, exog=None, date_train_end="2017-03-26", pred_days = [-28,39],\

                             ranges=(slice(1,3),slice(0,2),slice(1,3),  slice(0,2),slice(1,2),slice(1,2),slice(7,8)))

#Not shown, but the SMA parameter is important to keep to 1
mod = sm.tsa.statespace.SARIMAX(air.visit_mean[:450], trend='c', order=(1,0,2), seasonal_order=(1,1,1,7))

results = mod.fit()

#Predict on future data and on time periods already known for evaluation with RMSLE

pred = results.predict(start=450, end=516)

print(results.summary())

#Plot

plotSARIMAX(air.visit_mean, pred)
resbrute, gr=autoSARIMAX(endog=air.visit_mean, exog=hol.holiday_flg, date_train_end="2017-03-26", pred_days=[-28,39],\

                    ranges=(slice(1,3),slice(0,2),slice(1,3),  slice(0,2),slice(1,2),slice(1,2),slice(7,8)))
modx = sm.tsa.statespace.SARIMAX(air.visit_mean[:450], trend='c', exog=hol.holiday_flg[:450],\

                                 order=(1,0,2), seasonal_order=(1,1,1,7))

resultsx = modx.fit(start_params=0.1*np.ones( len(modx.param_terms)-2+ 2*2 ))

#Predict on future data and on time periods already known for evaluation with RMSLE

predx = resultsx.predict(start=450, end=516, exog=pd.DataFrame(hol.holiday_flg[450:]))

print(resultsx.summary())

plotSARIMAX(air.visit_mean, predx)
resbrute, gr = autoSARIMAX(endog=air.visit_mean, exog=hol, date_train_end="2017-03-26", pred_days=[-28,39],\

                    ranges=(slice(1,3),slice(0,2),slice(1,3),  slice(0,2),slice(1,2),slice(1,2),slice(7,8)))
modx2 = sm.tsa.statespace.SARIMAX(air.visit_mean[:450], trend='c', exog=hol[:450], order=(1,0,2), seasonal_order=(1,1,1,7))

resultsx2 = modx2.fit(start_params=0.1*np.ones( len(modx2.params_complete)))

#Predict on future data and on time periods already known for evaluation with RMSLE

predx2 = resultsx2.predict(start=450, end=516, exog=hol[450:])

print(resultsx2.summary())

plotSARIMAX(air.visit_mean, predx2)