import numpy as np

# for linear algebra and mathematical functions



import pandas as pd

# for dataframe manipulation



import statsmodels.api as sm

import statsmodels.formula.api

# for stats modeling



from statsmodels.regression.linear_model import OLS

# imports OLS so the residuals from the OLS model can be used in the Error Correction Model



full_data = pd.read_csv('../input/Oil Data.csv')

# loads the dataset



full_data.ebroad_r = np.log(full_data.TWEXBPA/99.84)

# generates a variable for the growth rate of the Real Trade Weighted U.S. Dollar Index: Broad, Goods 

# indexed to January 2000



full_data.emajor_r = np.log(full_data.TWEXMPA/98.189)

# generates a variable for the growth rate of the Real Trade Weighted U.S. Dollar Index: Major Currencies, 

# Goods indexed to January 2000



full_data.eoitp_r = np.log(full_data.TWEXOPA/110.784)

# generates a variable for the growth rate of the Real Trade Weighted U.S. Dollar Index: Other Important 

# Trading Partners, Goods indexed to January 2000



full_data.pus = np.log(full_data.Fred_CPIAUCNS)

# generates a variable for the growth rate of the Consumer Price Index for All Urban Consumers(All Items)



full_data.po = np.log(full_data.MCOILWTICO)

# generates a variable for the growth rate of the Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, 

# Oklahoma



full_data.rpo = full_data.po - full_data.pus

# generates a variable for the growth rate of rude Oil Prices: West Texas Intermediate (WTI) - Cushing,

# Oklahoma minus the growth rate of inflation aka Consumer Price Index for All Urban Consumers(All Items) 



full_data.Date = pd.to_datetime(full_data.Date)

full_data['month'] = full_data.Date.map(lambda x: x.month)

# creates a column for month



variables_to_keep = ['rpo', 'Date', 'month', 'ebroad_r', 'emajor_r', 'eoitp_r']

# creates a list of all the variables of interest



data = full_data[variables_to_keep]

# creates a new dataframe containing only the variables of interest



data =  pd.concat([data, pd.get_dummies(data.month, drop_first = True)], axis = 1)

# creates dummy variables for each month, dropping January to avoid multicollinearity



data.index = pd.DatetimeIndex(data.Date)

# sets the Date as the index



diff_data = data

diff_data['drpo'] = diff_data.rpo.diff()

diff_data['debroad_r'] = diff_data.ebroad_r.diff()

diff_data['demajor_r'] = diff_data.emajor_r.diff()

diff_data['deoitp_r'] = diff_data.eoitp_r.diff()

diff_data = diff_data.drop(columns = variables_to_keep)

# creates a dataframe of the differenced variables of interest where the value in time t is subtracted by

# the value at time t-1



data = data.drop(columns = ['Date', 'month', 'drpo', 'debroad_r', 'demajor_r', 'deoitp_r'])

# leaves only the non-differenced variables of interest indexed by the date



data.columns = ['rpo', 'ebroad_r', 'emajor_r', 'eoitp_r', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',

               'aug', 'sep', 'oct', 'nov', 'dec']

diff_data.columns = ['feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 

                    'drpo', 'debroad_r', 'demajor_r', 'deoitp_r']

# renames the month dummies for easier interpretation



data = data.sort_index()

diff_data = diff_data.sort_index()

# sorts both dataframes by their index to allow for trunctation based on the index



ebroad_r_model = OLS.from_formula(formula='rpo ~ ebroad_r+feb+mar+apr+may+jun+jul+aug+sep+oct+nov+dec',

                                  data = data)

ebroad_r_model_fit = ebroad_r_model.fit()

data['ebroad_r_resids'] = ebroad_r_model_fit.resid

diff_data['ebroad_r_resids'] = ebroad_r_model_fit.resid

# adds a column of the residuals from OLS regression of rpo on ebroad_r with seasonal effects



emajor_r_eoitp_r_model = OLS.from_formula(formula='''rpo ~ emajor_r+eoitp_r+feb+mar+apr+may+jun+jul+aug+sep+

                                            oct+nov+dec''', data = data)

emajor_r_eoitp_r_model_fit = emajor_r_eoitp_r_model.fit()

data['emajor_r_eoitp_r_resids'] = emajor_r_eoitp_r_model_fit.resid

diff_data['emajor_r_eoitp_r_resids'] = emajor_r_eoitp_r_model_fit.resid

# adds a column of the residuals from OLS regression of rpo on emajor_r and eoitp_r with seasonal effects



for i in ['rpo', 'ebroad_r', 'emajor_r', 'eoitp_r']:

    for j in range(1, 25):

        data.loc[:,i+"_"+str(j)] = data[i].shift(j)

# creates 24 lagged columns of the variables of interest

     

for i in ['drpo', 'debroad_r', 'demajor_r', 'deoitp_r']:

    for j in range(1, 25):

        diff_data.loc[:,i+"_"+str(j)] = diff_data[i].shift(j)

# creates 24 lagged columns of the variables of interest

diff_data['ebroad_resid_1L'] = diff_data.ebroad_r_resids.shift(1)

diff_data['comb_resid_1L'] = diff_data.emajor_r_eoitp_r_resids.shift(1)

diff_data = diff_data.drop('ebroad_r_resids', axis = 1)

diff_data = diff_data.drop('emajor_r_eoitp_r_resids', axis = 1)

# creates a lagged residual columns for error correction



lag_data = data.dropna(axis = 0)

lag_diff_data = diff_data.dropna(axis = 0)

# creats a dataframe without all the missing entries created by making lagged variables; the dataframe

# has data going from 2001-01o 2016-02



forecasting_data = data.truncate(before = '2016-03-01')

forecasting_diff_data = diff_data.truncate(before = '2016-03-01')

vec_data = data.truncate(before = '1999-01-01')

data = data.truncate(before = '1999-01-01', after = '2016-02-01')

diff_data = diff_data.truncate(before = '1999-01-01', after = '2016-02-01')

lag_data = lag_data.truncate(before = '1999-01-01', after = '2016-02-01')

lag_diff_data = lag_diff_data.truncate(before = '1999-01-01', after = '2016-02-01')

# cuts off the pre-Euro period and reserves 3 years of data for testing forecasting ability of the models



exog_vars_24L = lag_diff_data.columns.tolist()

exog_vars_24L.remove('drpo')

# creates a list for all the exogenous variables for the ECMs with 24 lags



exog_vars_12L = exog_vars_24L

for i in range(13, 25):

    exog_vars_12L = [ x for x in exog_vars_12L if str(i) not in x ]

# creates a list for all the exogenous variables for the ECMs with 12 lags

        

ebroad_ecm_exog_12L = exog_vars_12L 

for i in ['demajor_r', 'deoitp_r']:

    ebroad_ecm_exog_12L = [ x for x in ebroad_ecm_exog_12L if i not in x ]

    for j in range(1, 25):

        ebroad_ecm_exog_12L = [ x for x in ebroad_ecm_exog_12L if str(i) not in x ]

ebroad_ecm_exog_12L = [ x for x in ebroad_ecm_exog_12L if 'comb_resid_1L' not in x ]

# creates a list of vars for the ECM using ebroad_r with 12 lags

        

comb_ecm_exog_12L = exog_vars_12L 

for i in ['debroad_r']:

    comb_ecm_exog_12L = [ x for x in comb_ecm_exog_12L if i not in x ]

    for j in range(1, 25):

        comb_ecm_exog_12L = [ x for x in comb_ecm_exog_12L if str(i) not in x ]

comb_ecm_exog_12L = [ x for x in comb_ecm_exog_12L if 'ebroad_resid_1L' not in x ]

# creates a list of vars for the ECM using emajor_r and eoitp_r with 12 lags



non_diff_exog_vars_24L = lag_data.columns.tolist()

non_diff_exog_vars_24L.remove('rpo')

# creates a list for all the exogenous variables for the VECMs with 24 lags



non_diff_exog_vars_12L = non_diff_exog_vars_24L

for i in range(13, 25):

    non_diff_exog_vars_12L = [ x for x in non_diff_exog_vars_12L if str(i) not in x ]

# creates a list for all the exogenous variables for the VECMs with 12 lags



forecasting_diff_data_y = forecasting_diff_data.drpo

# creates a y variable for single equation error correction modeling



forecasting_data_ebroad_x_12L = forecasting_diff_data[ebroad_ecm_exog_12L]

forecasting_data_comb_x_12L = forecasting_diff_data[comb_ecm_exog_12L]

# creates exogenous variable dataframes for the forecasting data set when using error correction modelings
from matplotlib import pyplot

# imports pyplot from matplotlib for plotting functionality

from statsmodels.graphics.tsaplots import plot_acf

# imports the auto-correlation graphic function



for i in ['rpo', 'ebroad_r', 'emajor_r', 'eoitp_r']:

    plot_acf(data[i], lags = 15)

    pyplot.title('ACF for %s' % i) 

    pyplot.show()
for i in ['drpo', 'debroad_r', 'demajor_r', 'deoitp_r']:

    plot_acf(diff_data[i], lags = 15)

    pyplot.title('ACF for %s' % i) 

    pyplot.show()
from statsmodels.tsa.stattools import adfuller

# imports the Augmented Dickey-Fuller Test for establishing the order of integration of time series



for i in ['rpo', 'ebroad_r', 'emajor_r', 'eoitp_r']:

    for j in ['nc', 'c', 'ct', 'ctt']:

        result = adfuller(data[i], regression = j)

        print('ADF Statistic with %s for %s: %f' % (j, i, result[0]))

        print('p-value: %f' % result[1])

# performs the Augmented Dickey-Fuller Test for all our variables of interest without

# a constant, with a constant, with a constant and linear trend, and with a constant, linear trend, and

# quadratic trend
for i in ['drpo', 'debroad_r', 'demajor_r', 'deoitp_r']:

    for j in ['nc', 'c', 'ct', 'ctt']:

        result = adfuller(diff_data[i], regression = j)

        print('ADF Statistic with %s for %s: %f' % (j, i, result[0]))

        print('p-value: %f' % result[1])

# performs the Augmented Dickey-Fuller Test for all our variables of interest at 12 and 24 lags without a

# a constant, with a constant, with a constant and linear trend, and with a constant, linear trend, and

# quadratic trend
from statsmodels.tsa.stattools import grangercausalitytests

# imports test for Granger Causality (does variable x have value in predicting y)



# grangercausalitytests tests if the variable in the first column is Granger caused by the variable in the

# second column, so I made bivariate dataframes for each combination of rpo and real dollar exchange growth

# rates



rpo_ebroad_r = data[['rpo', 'ebroad_r']]

ebroad_r_rpo = data[['ebroad_r', 'rpo']]



results = grangercausalitytests(rpo_ebroad_r, maxlag = 15, addconst = True, verbose = True)

results = grangercausalitytests(ebroad_r_rpo, maxlag = 15, addconst = True, verbose = True)
rpo_emajor_r = data[['rpo', 'emajor_r']]

emajor_r_rpo = data[['emajor_r', 'rpo']]



results = grangercausalitytests(rpo_emajor_r, maxlag = 15, addconst = True, verbose = True)

results = grangercausalitytests(emajor_r_rpo, maxlag = 15, addconst = True, verbose = True)
rpo_eoitp_r = data[['rpo', 'eoitp_r']]

eoitp_r_rpo = data[['eoitp_r', 'rpo']]



results = grangercausalitytests(rpo_eoitp_r, maxlag = 15, addconst = True, verbose = True)

results = grangercausalitytests(eoitp_r_rpo, maxlag = 15, addconst = True, verbose = True)
from statsmodels.tsa.stattools import coint

# imports coint which is tests if the residuals of OLS estimates of models are stationary, which is a way

# of saying that they are cointegrated



result = coint(data.rpo, data.ebroad_r, trend='c', method='aeg', maxlag=24, autolag='aic', 

                return_results=None)



print('p-value for Cointegration between rpo and ebroad_r: %f' % result[1])



result = coint(data.rpo, data[['emajor_r', 'eoitp_r']], trend='c', method='aeg', maxlag=24, autolag='aic', 

                return_results=None)



print('p-value for Cointegration between rpo and emajor_r & eoitp_r: %f' % result[1])
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

# for testing for serial correlation in residuals using the Breusch-Godfrey Test

from statsmodels.stats.diagnostic import het_breuschpagan

# for testing for heteroskedasticity in the residuals using the Breusch-Pagan Test

from statsmodels.stats.diagnostic import normal_ad

# for testing for normality in the residuals using the Andersson-Darling Test



print('12-Lag ECM: drpo on lagged drpo and ebroad_r')

ecm_ebroad_12L = OLS(endog = lag_diff_data.drpo, exog = lag_diff_data[ebroad_ecm_exog_12L])

ecm_ebroad_12L_fit = ecm_ebroad_12L.fit(cov_type='HC0')

print(ecm_ebroad_12L_fit.summary())

print()

results_acorr = acorr_breusch_godfrey(ecm_ebroad_12L_fit, nlags=None, store=False)

print('Bruesch-Godfrey Test for serial correlation p-value: %f' % results_acorr[3])

results_het = het_breuschpagan(ecm_ebroad_12L_fit.resid, lag_diff_data[ebroad_ecm_exog_12L])

print('Bruesch-Pagan Test for heteroskedasticity p-value: %f' % results_het[3])

results_norm = normal_ad(ecm_ebroad_12L_fit.resid, axis=0)

print('Anderson-Darling Test for normality p-value: %f' % results_norm[1])

print()

print()

print('12-Lag ECM: drpo on lagged drpo, emajor_r, and eoitp_r')

ecm_comb_12L = OLS(endog = lag_diff_data.drpo, exog = lag_diff_data[comb_ecm_exog_12L])

ecm_comb_12L_fit = ecm_comb_12L.fit(cov_type='HC0')

print(ecm_comb_12L_fit.summary())

print()

results_acorr = acorr_breusch_godfrey(ecm_comb_12L_fit, nlags=None, store=False)

print('Bruesch-Godfrey Test for serial correlation p-value: %f' % results_acorr[3])

results_het = het_breuschpagan(ecm_comb_12L_fit.resid, lag_diff_data[comb_ecm_exog_12L])

print('Bruesch-Pagan Test for heteroskedasticity p-value: %f' % results_het[3])

results_norm = normal_ad(ecm_comb_12L_fit.resid, axis=0)

print('Anderson-Darling Test for normality p-value: %f' % results_norm[1])
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# imports a function to create prediction intervals for the graphs

from statsmodels.tools.eval_measures import rmse

# imports the root mean square error function to compare forecast accuracy



preds_ebroad_12L = ecm_ebroad_12L_fit.predict(forecasting_data_ebroad_x_12L)

preds_comb_12L = ecm_comb_12L_fit.predict(forecasting_data_comb_x_12L)



fig, ax = pyplot.subplots(figsize=(14,5))

ax.plot(forecasting_diff_data_y.index, forecasting_diff_data_y, label = "Data")

sdev, lower, upper = wls_prediction_std(ecm_ebroad_12L_fit, exog=forecasting_data_ebroad_x_12L, alpha=0.05)

ax.fill_between(forecasting_diff_data_y.index, lower, upper, color='#888888', alpha=0.05)

ax.plot(forecasting_diff_data_y.index, ecm_ebroad_12L_fit.predict(forecasting_data_ebroad_x_12L), 'r', 

        label="ECM Prediction")

fig.suptitle('ECM using ebroad_r at 12 lags w/ 95% Confidence Interval', fontsize=20)

ax.legend(loc="best");

RMSE = rmse(preds_ebroad_12L, forecasting_diff_data_y)

print('RMSE for the ECM using ebroad_r at 12 lags: %f' % RMSE)

print()



fig, ax = pyplot.subplots(figsize=(14,5))

ax.plot(forecasting_diff_data_y.index, forecasting_diff_data_y, label = "Data")

sdev, lower, upper = wls_prediction_std(ecm_comb_12L_fit, exog=forecasting_data_comb_x_12L, alpha=0.05)

ax.fill_between(forecasting_diff_data_y.index, lower, upper, color='#888888', alpha=0.05)

ax.plot(forecasting_diff_data_y.index, ecm_comb_12L_fit.predict(forecasting_data_comb_x_12L), 'r', 

        label="ECM Prediction")

fig.suptitle('ECM using emajor_r and eoitp_r at 12 lags w/ 95% Confidence Interval', fontsize=20)

ax.legend(loc="best");

RMSE = rmse(preds_comb_12L, forecasting_diff_data_y)

print('RMSE for the ECM using emajor_r and eoitp_r at 12 lags: %f' % RMSE)

print()
from statsmodels.tsa.vector_ar.vecm import coint_johansen



# uses https://nbviewer.jupyter.org/github/mapsa/seminario-doc-2014/blob/master/cointegration-example.ipynb

# to create functions to return the number of cointegrating vectors based on the Trace Test

def johansen_trace(y, p):

        N, l = y.shape

        joh_trace = coint_johansen(y, 0, p)

        r = 0

        for i in range(l):

            if joh_trace.lr1[i] > joh_trace.cvt[i, 2]:     # 0: 90%  1:95% 2: 99%

                r = i + 1

        joh_trace.r = r



        return joh_trace



for i in [3, 6, 9, 12, 15, 18, 21, 24]: # tests on a quarterly basis

    joh_trace = johansen_trace(data[['rpo', 'emajor_r']], i)

    print('Using the Trace Test, there are', joh_trace.r, '''cointegration vectors at %s lags between

    rpo and ebroad_r''' % i)

    print()



for i in [3, 6, 9, 12, 15, 18, 21, 24]: # tests on a quarterly basis

    joh_trace = johansen_trace(data[['rpo', 'emajor_r', 'eoitp_r']], i)

    print('Using the Trace Test, there are', joh_trace.r, '''cointegration vectors at %s lags between

    rpo, emajor_r, and eoitp_r''' % i)

    print()
from statsmodels.tsa.vector_ar.vecm import VECM



VECM_ebroad = VECM(endog = vec_data[['rpo', 'ebroad_r']], k_ar_diff=6, coint_rank=1, deterministic='ci',

                  seasons=12, first_season=1)

VECM_ebroad_fit = VECM_ebroad.fit()

VECM_ebroad_fit.summary()
irf = VECM_ebroad_fit.irf(50)

irf.plot(orth=False)
VECM_ebroad_fit.plot_forecast(36)
VECM_comb = VECM(endog = vec_data[['rpo', 'emajor_r', 'eoitp_r']], k_ar_diff=9, coint_rank=1, 

                   deterministic='ci', seasons=12, first_season=1)

VECM_comb_fit = VECM_comb.fit()

VECM_comb_fit.summary()
irf = VECM_comb_fit.irf(50)

irf.plot(orth=False)
VECM_comb_fit.plot_forecast(36)