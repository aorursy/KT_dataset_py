import numpy as np

import pandas as pd

import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller, kpss, acf, grangercausalitytests

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf,month_plot,quarter_plot

from scipy import signal

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline 

sns.set_style("whitegrid")

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15) 
d = pd.read_csv('../input/m5-data-for-tsa/data_for_tsa.csv')

d.head()
fig, ax = plt.subplots(figsize=(15, 6))

ax.set_title('Demand over Time', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax.set_xlabel('Time', fontsize = 16, fontdict=dict(weight='bold'))

ax.set_ylabel('Demand', fontsize = 16, fontdict=dict(weight='bold'))

plt.tick_params(axis='y', which='major', labelsize=16);plt.tick_params(axis='x', which='major', labelsize=16)

d.plot(x='date',y='demand',figsize=(15,5),ax=ax);
variable = 'demand'

fig, ax = plt.subplots(figsize=(15, 6))



palette = sns.color_palette("colorblind", 6)

sns.lineplot(d['month'], d[variable], hue=d['year'], palette=palette)

ax.set_title('Seasonal plot of demand', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax.set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'));ax.set_ylabel('Demand', fontsize = 16, fontdict=dict(weight='bold'))



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))



sns.boxplot(d['year'], d[variable], ax=ax[0])

ax[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax[0].set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'));ax[0].set_ylabel('Demand', fontsize = 16, fontdict=dict(weight='bold'))



sns.boxplot(d['month'], d[variable], ax=ax[1])

ax[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax[1].set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'));ax[1].set_ylabel('Demand', fontsize = 16, fontdict=dict(weight='bold'));
from pylab import rcParams

rcParams['figure.figsize'] = 15, 12

rcParams['axes.labelsize'] = 20

rcParams['ytick.labelsize'] = 16

rcParams['xtick.labelsize'] = 16



y = d[['date','demand']].set_index('date')

y = y.asfreq('d')

decomposition = sm.tsa.seasonal_decompose(y, model='additive')

decomp = decomposition.plot()

decomp.suptitle('Demand Decomposition', fontsize=22);
from statsmodels.tsa.stattools import adfuller

# check for stationarity

def adf_test(series,title=''):

    """

    Pass in a time series and an optional title, returns an ADF report

    """

    print('Augmented Dickey-Fuller Test: {}'.format(title))

    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data

    

    labels = ['ADF test statistic','p-value','# lags used','# observations']

    out = pd.Series(result[0:4],index=labels)



    for key,val in result[4].items():

        out['critical value ({})'.format(key)]=val

        

    print(out.to_string())          # .to_string() removes the line "dtype: float64"

    

    if result[1] <= 0.05:

        print("Strong evidence against the null hypothesis")

        print("Reject the null hypothesis")

        print("Data has no unit root and is stationary")

    else:

        print("Weak evidence against the null hypothesis")

        print("Fail to reject the null hypothesis")

        print("Data has a unit root and is non-stationary")
adf_test(d[['date','demand']]['demand'],title='Demand')
from statsmodels.tsa.statespace.tools import diff



fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(15, 11))



d['demand_Diff1'] = diff(d['demand'],k_diff=1)

d['demand_Diff2'] = diff(d['demand'],k_diff=2)

d['demand_Diff3'] = diff(d['demand'],k_diff=3)



d['demand'].plot(title="Initial Data",ax=ax[0][0]).autoscale(axis='x',tight=True);

d['demand_Diff1'].plot(title="First Difference Data",ax=ax[0][1]).autoscale(axis='x',tight=True);

d['demand_Diff2'].plot(title="Second Difference Data",ax=ax[1][0]).autoscale(axis='x',tight=True);

d['demand_Diff3'].plot(title="Third Difference Data",ax=ax[1][1]).autoscale(axis='x',tight=True);
adf_test(d[['date','demand_Diff1']]['demand_Diff1'],title='demand_Diff1')
from pandas.plotting import lag_plot

lag_plot(d['demand']);
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15, 6))

autocorr = acf(d['demand'], nlags=30, fft=False) # just the numbers

plot_acf(d['demand'].tolist(), lags=30, ax=ax[0]); # just the plot

plot_pacf(d['demand'].tolist(), lags=30, ax=ax[1]); # just the plot