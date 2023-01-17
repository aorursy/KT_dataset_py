import os

import pandas as pd

import numpy as np

import random

import itertools

from arch import arch_model

from scipy.stats import shapiro

from scipy.stats import probplot

from statsmodels.stats.diagnostic import het_arch

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.stats.diagnostic import acorr_ljungbox



from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight') 

plt.rcParams['xtick.labelsize'] = 10

plt.rcParams['ytick.labelsize'] = 10

%matplotlib inline
path = '../input/individual_stocks_5yr/individual_stocks_5yr'

csvs = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]



df = pd.DataFrame()

for file in random.sample(range(1, len(csvs)), 8):

    stock_df = pd.read_csv(csvs[file])

    stock_df.index = pd.DatetimeIndex(stock_df.date)

    name = stock_df['Name'].iloc[0]

    df[name] = stock_df['close']



df.plot(figsize=(10, 5), title='Closing Price for 8 Random Stocks')

df.head()
stock = 'TDG'

df = pd.read_csv(f'../input/individual_stocks_5yr/individual_stocks_5yr/{stock}_data.csv')

df.index = pd.DatetimeIndex(df.date)

df = df.drop(columns=['open', 'high', 'low', 'volume', 'date', 'Name'])

df['pct_change'] = 100*df['close'].pct_change()

df.dropna(inplace=True)

df['close'].plot(figsize=(10, 5), title=f'{stock} Closing Price 2013-2018')

plt.show()

df['pct_change'].plot(figsize=(10, 5), title=f'{stock} Percent Change in Closing Price')

plt.show()

acf = plot_acf(df['pct_change'], lags=30)

pacf = plot_pacf(df['pct_change'], lags=30)

acf.suptitle(f'{stock} Percent Change Autocorrelation and Partial Autocorrelation', fontsize=20)

acf.set_figheight(5)

acf.set_figwidth(15)

pacf.set_figheight(5)

pacf.set_figwidth(15)

plt.show()
ljung_res = acorr_ljungbox(df['pct_change'], lags= 40, boxpierce=True)

print(f'Ljung-Box p-values: {ljung_res[1]}')

print(f'\nBox-Pierce p-values: {ljung_res[3]}')
def ts_plot(residuals, stan_residuals, lags=50):

    residuals.plot(title='GARCH Residuals', figsize=(15, 10))

    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    ax[0].set_title('GARCH Standardized Residuals KDE')

    ax[1].set_title('GARCH Standardized Resduals Probability Plot')    

    residuals.plot(kind='kde', ax=ax[0])

    probplot(stan_residuals, dist='norm', plot=ax[1])

    plt.show()

    acf = plot_acf(stan_residuals, lags=lags)

    pacf = plot_pacf(stan_residuals, lags=lags)

    acf.suptitle('GARCH Model Standardized Residual Autocorrelation', fontsize=20)

    acf.set_figheight(5)

    acf.set_figwidth(15)

    pacf.set_figheight(5)

    pacf.set_figwidth(15)

    plt.show()
garch = arch_model(df['pct_change'], vol='GARCH', p=1, q=1, dist='normal')

fgarch = garch.fit(disp='off') 

resid = fgarch.resid

st_resid = np.divide(resid, fgarch.conditional_volatility)

ts_plot(resid, st_resid)

fgarch.summary()
arch_test = het_arch(resid, maxlag=50)

shapiro_test = shapiro(st_resid)



print(f'Lagrange mulitplier p-value: {arch_test[1]}')

print(f'F test p-value: {arch_test[3]}')

print(f'Shapiro-Wilks p-value: {shapiro_test[1]}')
def gridsearch(data, p_rng, q_rng):

    top_score, top_results = float('inf'), None

    top_models = []

    for p in p_rng:

        for q in q_rng:

            try:

                model = arch_model(data, vol='GARCH', p=p, q=q, dist='normal')

                model_fit = model.fit(disp='off')

                resid = model_fit.resid

                st_resid = np.divide(resid, model_fit.conditional_volatility)

                results = evaluate_model(resid, st_resid)

                results['AIC'] = model_fit.aic

                results['params']['p'] = p

                results['params']['q'] = q

                if results['AIC'] < top_score: 

                    top_score = results['AIC']

                    top_results = results

                elif results['LM_pvalue'][1] is False:

                    top_models.append(results)

            except:

                continue

    top_models.append(top_results)

    return top_models

                

def evaluate_model(residuals, st_residuals, lags=50):

    results = {

        'LM_pvalue': None,

        'F_pvalue': None,

        'SW_pvalue': None,

        'AIC': None,

        'params': {'p': None, 'q': None}

    }

    arch_test = het_arch(residuals, maxlag=lags)

    shap_test = shapiro(st_residuals)

    # We want falsey values for each of these hypothesis tests

    results['LM_pvalue'] = [arch_test[1], arch_test[1] < .05]

    results['F_pvalue'] = [arch_test[3], arch_test[3] < .05]

    results['SW_pvalue'] = [shap_test[1], shap_test[1] < .05]

    return results
p_rng = range(0,30)

q_rng = range(0,40)

df['dif_pct_change'] = df['pct_change'].diff()

top_models = gridsearch(df['dif_pct_change'], p_rng, q_rng)

print(top_models)
garch = arch_model(df['pct_change'], vol='GARCH', p=17, q=25, dist='normal')

fgarch = garch.fit(disp='off') 

resid = fgarch.resid

st_resid = np.divide(resid, fgarch.conditional_volatility)

ts_plot(resid, st_resid)

arch_test = het_arch(resid, maxlag=50)

shapiro_test = shapiro(st_resid)

print(f'Lagrange mulitplier p-value: {arch_test[1]}')

print(f'F test p-value: {arch_test[3]}')

print(f'Shapiro-Wilks p-value: {shapiro_test[1]}')

fgarch.summary()