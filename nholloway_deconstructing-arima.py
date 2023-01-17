import math

import itertools

import os

import pandas as pd

import numpy as np

import operator

from collections import defaultdict

from scipy.stats import boxcox, shapiro, probplot, jarque_bera

from sklearn.metrics import mean_squared_error, mean_absolute_error 

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.arima_model import ARIMAResults

from statsmodels.stats.diagnostic import acorr_ljungbox

from statsmodels.stats.diagnostic import acorr_breusch_godfrey



from statsmodels.tsa.stattools import adfuller, kpss

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight') 

plt.rcParams['xtick.labelsize'] = 20

plt.rcParams['ytick.labelsize'] = 20

%matplotlib inline
path = '../input/csvs_per_year/csvs_per_year'

files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

df = pd.concat((pd.read_csv(file) for file in files), sort=False)

df = df.groupby(['date']).agg('mean')

df.index = pd.DatetimeIndex(data= df.index)
col_list = ['BEN', 'CO', 'EBE', 'NMHC', 'NO_2', 'O_3', 'PM10', 'SO_2', 'TCH', 'TOL']

pm10_df = pd.DataFrame(df['PM10'].resample('M').mean())

pm10_df.rename(columns= {'PM10':'pm10'}, inplace=True)

pm10_df.plot(figsize=(15, 6))

plt.title('PM10 in Madrid Air from 2001-2019', fontsize=20)

plt.legend(loc='upper left')

plt.show()
def adf_test(timeseries):

    print ('Results of Dickey-Fuller Test:')

    print('Null Hypothesis: Unit Root Present')

    print('Test Statistic < Critical Value => Reject Null')

    print('P-Value =< Alpha(.05) => Reject Null\n')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput[f'Critical Value {key}'] = value

    print (dfoutput, '\n')



def kpss_test(timeseries, regression='c'):

    # Whether stationary around constant 'c' or trend 'ct

    print ('Results of KPSS Test:')

    print('Null Hypothesis: Data is Stationary/Trend Stationary')

    print('Test Statistic > Critical Value => Reject Null')

    print('P-Value =< Alpha(.05) => Reject Null\n')

    kpsstest = kpss(timeseries, regression=regression)

    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])

    for key,value in kpsstest[3].items():

        kpss_output[f'Critical Value {key}'] = value

    print (kpss_output, '\n')

pm10_df['bc_pm10'], lamb = boxcox(pm10_df.pm10)

pm10_df['d1_pm10'] = pm10_df['bc_pm10'].diff()

pm10_df['d2_pm10'] = pm10_df['d1_pm10'].diff()

pm10_df['d3_pm10'] = pm10_df['d2_pm10'].diff()

fig = plt.figure(figsize=(20,40))



plt_bc = plt.subplot(411)

plt_bc.plot(pm10_df.bc_pm10)

plt_bc.title.set_text('Box-Cox Transform')

plt_d1 = plt.subplot(412)

plt_d1.plot(pm10_df.d1_pm10)

plt_d1.title.set_text('First-Order Transform')

plt_d2 = plt.subplot(413)

plt_d2.plot(pm10_df.d2_pm10)

plt_d2.title.set_text('Second-Order Transform')

plt_d3 = plt.subplot(414)

plt_d3.plot(pm10_df.d3_pm10)

plt_d3.title.set_text('Third-Order Transform')

plt.show()



pm10_df.bc_pm10.dropna(inplace=True)

pm10_df.d1_pm10.dropna(inplace=True)

pm10_df.d2_pm10.dropna(inplace=True)



print('Unit Root Tests:')

print('BoxCox-No Difference')

adf_test(pm10_df.bc_pm10)

kpss_test(pm10_df.bc_pm10)

print('\nFirst Difference:')

adf_test(pm10_df.d1_pm10)

kpss_test(pm10_df.d1_pm10)

f_acf = plot_acf(pm10_df['d2_pm10'], lags=50)

f_pacf = plot_pacf(pm10_df['d2_pm10'], lags=50)

f_acf.set_figheight(10)

f_acf.set_figwidth(15)

f_pacf.set_figheight(10)

f_pacf.set_figwidth(15)

plt.show()
def evaluate_model(data, pdq):

    split_date = '2012-01-01'

    train, test = data[:split_date], data[split_date:]

    test_len = len(test)

    model = ARIMA(train, order=pdq)

    model_fit = model.fit(disp=-1)

    predictions = model_fit.forecast(test_len)

    aic = model_fit.aic

    mse = mean_squared_error(test, predictions[0])

    rmse = math.sqrt(mse)

    return {'rmse': rmse, 'aic': aic}





def gridsearch(data, p_range, d_range, q_range):

    models = defaultdict()

    best_score, best_params = float('inf'), None

    for p in p_range:

        for d in d_range:

            for q in q_range:

                params = (p,d,q)

                try:

                    score = evaluate_model(data, params)

                    models[str(params)] = score

                    if score['aic'] < best_score:

                        best_score, best_params = score['aic'], params

                except:

                    continue

    return best_params, models

p_rng = range(0, 10)

d_rng = [2]

q_rng = range(0, 10)



parameter, models = gridsearch(pm10_df['bc_pm10'], p_rng, d_rng, q_rng)
sorted_scores = sorted(models.items(), key = lambda x:x[1]['aic'])



print('Best ARIMA Parameters and Scores Ranked By AIC')

for i in range(6):

    print(f'{sorted_scores[i][0]}, {sorted_scores[i][1]}')
def model_diagnostics(residuals, model_obj):

    # For Breusch-Godfrey we have to pass the results object

    godfrey = acorr_breusch_godfrey(model_obj, nlags= 40)

    ljung = acorr_ljungbox(residuals, lags= 40)

    shap = shapiro(residuals)

    j_bera = jarque_bera(residuals)

    print('Results of Ljung-Box:')

    print('Null Hypothesis: No auotcorrelation')

    print('P-Value =< Alpha(.05) => Reject Null')

    print(f'p-values: {ljung[1]}\n')

    print('Results of Breusch-Godfrey:')

    print('Null Hypothesis: No auotcorrelation')

    print('P-Value =< Alpha(.05) => Reject Null')   

    print(f'p-values: {godfrey[1]}\n')

    print('Results of Shapiro-Wilks:')

    print('Null Hypothesis: Data is normally distributed')

    print('P-Value =< Alpha(.05) => Reject Null')   

    print(f'p-value: {shap[1]}\n')

    print('Results of Jarque-Bera:')

    print('Null Hypothesis: Data is normally distributed')

    print('P-Value =< Alpha(.05) => Reject Null')   

    print(f'p-value: {j_bera[1]}')



def plot_diagnostics(residuals):

    residuals.plot(title='ARIMA Residuals', figsize=(15, 10))

    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    ax[0].set_title('ARIMA Residuals KDE')

    ax[1].set_title('ARIMA Resduals Probability Plot')    

    residuals.plot(kind='kde', ax=ax[0])

    probplot(residuals, dist='norm', plot=ax[1])

    plt.show()  
best_parameters = (5, 2, 1)

model = ARIMA(pm10_df['bc_pm10'], order=best_parameters)

model_fit = model.fit(disp=-1)

resid = model_fit.resid



model_diagnostics(resid, model_fit)

plot_diagnostics(resid)
best_parameters = (5, 2, 1)

split_date = '2012-01-01'

data = pm10_df['bc_pm10']

train, test = data[:split_date], data[split_date:]

test_len = len(test)

model = ARIMA(train, order=best_parameters)

model_fit = model.fit(disp=-1)

prediction = model_fit.forecast(test_len)



pred_df = pd.DataFrame(prediction[0], index= test.index)

mae = mean_absolute_error(test, pred_df)

mse = mean_squared_error(test, pred_df)

rmse = math.sqrt(mse)



plt.figure(figsize=(20, 10))

plt.title('How ARIMA Fits Volatile Data', fontsize=30)

plt.plot(train, label='Train')

plt.plot(pred_df, label='Prediction')

plt.plot(test, label='Test')



print(f'Mean Squared Error: {mse}')

print(f'Root Mean Squared Error: {rmse}')

print(f'Mean Absolute Error: {mae}')

plt.legend(fontsize= 25)

plt.show()