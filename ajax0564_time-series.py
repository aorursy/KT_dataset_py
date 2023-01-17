import requests

import statsmodels.api as sm

import io

import pandas as pd

# Load Dataset

DATA_URL="http://robjhyndman.com/tsdldata/data/nybirths.dat"

fopen = requests.get(DATA_URL).content

ds=pd.read_csv(io.StringIO(fopen.decode('utf-8')), header=None,

names=['birthcount'])

print(ds.head())

# Add time index

date=pd.date_range("1946-01-01", "1959-12-31", freq="1M")

ds['Date']=pd.DataFrame(date)

ds = ds.set_index('Date')

# decompose dataset

res = sm.tsa.seasonal_decompose(ds.birthcount, model="additive")

resplot = res.plot()
ds.head(3)
import matplotlib.pyplot as plt

# Function for Single exponential smoothing

def single_exp_smoothing(x, alpha):

    F = [x[0]] # first value is same as series

    for t in range(1, len(x)):

        F.append(alpha * x[t] + (1 - alpha) * F[t-1])

    return pd.DataFrame(F)



alpha1 = single_exp_smoothing(ds['birthcount'],0.2).set_index(ds.index)

plt.figure(figsize=(20,10))

           

plt.plot(alpha1,label = "alpha = {}".format(0.2))

plt.plot(ds['birthcount'],label = 'original')

plt.plot(single_exp_smoothing(ds['birthcount'],0.4).set_index(ds.index),label = 'alpha = {}'.format(0.4))

plt.plot(single_exp_smoothing(ds['birthcount'],0.6).set_index(ds.index),label = 'alpha = {}'.format(0.6))

plt.plot(single_exp_smoothing(ds['birthcount'],0.8).set_index(ds.index),label = 'alpha = {}'.format(0.8))

plt.plot(single_exp_smoothing(ds['birthcount'],1).set_index(ds.index),label = 'alpha = {}'.format(1))

plt.legend()

plt.grid()
plt.figure(figsize = (10,10))

plt.plot(single_exp_smoothing(ds['birthcount'],0.6).set_index(ds.index),label = 'alpha = {}'.format(0.8))

plt.plot(ds['birthcount'],label = 'original')

plt.legend()

plt.grid()
# Function for double exponential smoothing

def double_exp_smoothing(x, alpha, beta):

    yhat = [x[0]] # first value is same as series

    for t in range(1, len(x)):

        if t==1:

            F, T= x[0], x[1] - x[0]

        F_n_1, F = F, alpha*x[t] + (1-alpha)*(F+T)

        T=beta*(F-F_n_1)+(1-beta)*T

        yhat.append(F+T)

    return pd.DataFrame(yhat)



plt.figure(figsize = (10,10))

plt.plot(ds['birthcount'],label = 'original')

plt.plot(double_exp_smoothing(ds['birthcount'],0.2,0.2).set_index(ds.index),label = '(a,b) = ({},{})'.format(0.2,0.2))

plt.plot(double_exp_smoothing(ds['birthcount'],0.6,0.7).set_index(ds.index),label = '(a,b) = ({},{})'.format(0.6,0.7))

plt.plot(double_exp_smoothing(ds['birthcount'],1,1).set_index(ds.index),label = '(a,b) = ({},{})'.format(1,1))

plt.legend()

plt.grid()
# Initialize trend value

def initialize_T(x, seasonLength):

    total=0.0

    for i in range(seasonLength):

         total+=float(x[i+seasonLength]-x[i])/seasonLength

    return total



# Initialize seasonal trend

def initialize_seasonalilty(x, seasonLength):

    seasons={}

    seasonsMean=[]

    num_season=int(len(x)/seasonLength)

# Compute season average

    for i in range(num_season):

        seasonsMean.append(sum(x[seasonLength*i:seasonLength*i+seasonLength])/float(seasonLength))

# compute season intial values

    for i in range(seasonLength):

        tot=0.0

        for j in range(num_season):

            tot+=x[seasonLength*j+i]-seasonsMean[j]

        seasons[i]=tot/num_season

    return seasons
# Triple Exponential Smoothing Forecast

def triple_exp_smoothing(x, seasonLength, alpha, beta, gamma, h):

    yhat=[]

    S = initialize_seasonalilty(x, seasonLength)

    for i in range(len(x)+h):

        if i == 0:

            F = x[0]

            T = initialize_T(x, seasonLength)

            yhat.append(x[0])

            continue

        if i >= len(x):

              m = i - len(x) + 1

              yhat.append((F + m*T) + S[i%seasonLength])

        else:

            obsval = x[i]

            F_last, F= F, alpha*(obsval-S[i%seasonLength]) + (1-alpha)*(F+T)

            T = beta * (F-F_last) + (1-beta)*T

            S[i%seasonLength] = gamma*(obsval-F) + (1-gamma)*S[i%seasonLength]

            yhat.append(F+T+S[i%seasonLength])

    return pd.DataFrame(yhat)
plt.figure(figsize = (10,10))

plt.plot(ds['birthcount'],label = 'original')



plt.plot(triple_exp_smoothing(ds['birthcount'], 12, 0, 0,1, 0).set_index(ds.index),label = '{},{},{}'.format(0,0,1))

plt.plot(triple_exp_smoothing(ds['birthcount'], 12, 0.5, 0.5,1, 0).set_index(ds.index),label = '{},{},{}'.format(0.5,0.5,1))

plt.plot(triple_exp_smoothing(ds['birthcount'], 12, 0.6, 0.6,0.6, 0).set_index(ds.index),label = '{},{},{}'.format(0.6,0.6,0.6))

plt.grid()

plt.legend()
from statsmodels.tsa import seasonal

addetive_model = seasonal.seasonal_decompose(ds['birthcount'].tolist(),period=12, model='additive')

multiplactive_model = seasonal.seasonal_decompose(ds['birthcount'].tolist(),period=12, model='multiplicative')
addetive_model.plot()
multiplactive_model = seasonal.seasonal_decompose(ds['birthcount'].tolist(),period=12, model='multiplicative')

multiplactive_model.plot()
ds.head(3)
ds['Date'] = ds.index

ds.head(3)
sample = ds['birthcount'].resample('M')

sample.mean()
import statsmodels.tsa.api as smtsa



ar1model = smtsa.ARMA(ds.birthcount.tolist(), order=(1, 0))

ar1=ar1model.fit(maxlag=30, method='mle', trend='nc')

ar1.summary()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

ds['residual']=ds.birthcount-ds.birthcount.mean()

ds_df=ds.dropna()

plot_acf(ds_df.residual, lags=50)

plot_pacf(ds_df.residual, lags=50)
data = ds.birthcount-ds.birthcount.shift()
# To get the optimal p and q orders for ARMA, a grid search is performed with AIC

# minimization as the search criteria using the following script:

# # Optimize ARMA parameters

aicVal=[]

for ari in range(1, 3):

    for maj in range(1,3):

        arma_obj = smtsa.ARMA(data.dropna(), order=(ari,maj)).fit(maxlag=30, method='mle', trend='nc',transparams=False)

        aicVal.append([ari, maj, arma_obj.aic])
pd.DataFrame(aicVal)
# Building optimized model using minimum AIC

arma_obj_fin = smtsa.ARMA(ds.birthcount.tolist(), order=(1,2)).fit(maxlag=30, method='mle', trend='nc')

ds['ARMA12']=arma_obj_fin.predict()
arma_obj_fin = smtsa.ARMA(ds.birthcount.tolist(), order=(2,0)).fit(maxlag=30, method='mle', trend='nc')

ds['ARMA20']=arma_obj_fin.predict()
ds.head(3)
plt.figure(figsize = (10,10))

plt.plot(ds['birthcount'].iloc[1:],label = 'original')

plt.plot(ds['ARMA12'].iloc[1:],label = 'arma12')

plt.plot(ds['ARMA20'].iloc[1:],label = 'arma20')

plt.legend()
import seaborn as sns

sns.regplot(ds.birthcount,ds.ARMA12)

plt.grid()
sns.regplot(ds.birthcount,ds.ARMA20)

plt.grid()
ds.birthcount.shape
mean1, mean2 =ds.iloc[:84].birthcount.mean(),ds.iloc[84:].birthcount.mean()

var1, var2 = ds.iloc[:84].birthcount.var(), ds.iloc[84:].birthcount.var()

print('mean1=%f, mean2=%f' % (mean1, mean2))

print('variance1=%f, variance2=%f' % (var1, var2))
# ADF Test

from statsmodels.tsa.stattools import adfuller

adf_result= adfuller(ds.birthcount.tolist())

print('ADF Statistic: %f' % adf_result[0])

print('p-value: %f' % adf_result[1])
#Let us plot the original time series and first-differences

first_order_diff = ds['birthcount'].diff(1)

fig, ax = plt.subplots(2, sharex=True)

fig.set_size_inches(5.5, 5.5)

ds['birthcount'].plot(ax=ax[0], color='b')

ax[0].set_title('birthcount')

first_order_diff.plot(ax=ax[1], color='r')

ax[1].set_title('First-order differences ')
aicVal=[]

d = 1

for d in range(0,3):

    for ari in range(0, 3):

        for maj in range(0,3):

            try:

            

                 arma_obj = smtsa.ARIMA(data.dropna(), order=(ari,d,maj)).fit()

                 aicVal.append([ari,d, maj, arma_obj.aic])

            

            except ValueError:

                pass



pd.DataFrame(aicVal)
# Evaluating fit using optimal parameter

import numpy as np

arima_obj = smtsa.ARIMA(ds.birthcount.tolist(), order=(2,0,2))

arima_obj_fit = arima_obj.fit(disp=0)

arima_obj_fit.summary()

# Evaluate prediction

pred = arima_obj_fit.fittedvalues.tolist()
ds['ARIMA']=pred

# diffval=np.append([0,24], arima_obj_fit.resid+arima_obj_fit.fittedvalues)

# ds['diffval']=diffval

# The comparison with the actual and forecasted values is obtained and visualized using the

# following script:

# # Plot the curves

f, axarr = plt.subplots(1, sharex=True)

f.set_size_inches(5.5, 5.5)

ds['ARIMA'].plot(color='r', linestyle = '--', ax=axarr)

ds['birthcount'].plot(color='b')

axarr.set_title('ARIMA(2,0,2)')

plt.xlabel('Index')

plt.ylabel('birthcount')
import seaborn as sns

sns.regplot(ds.birthcount,ds.ARIMA)

plt.grid()
f, err, ci=arima_obj_fit.forecast(40)

plt.plot(f)

plt.plot(ci)

plt.grid()

plt.xlabel('Forecasting Index')

plt.ylabel('Forecasted value')