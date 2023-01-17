# Importing Libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline    

import warnings

warnings.filterwarnings('ignore')

plt.style.use('Solarize_Light2') 

from pylab import rcParams  



import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import cufflinks
# Processing the Data



df = pd.read_csv('../input/prc_hicp.csv')

df.head(2)
df.drop(['GEO', 'UNIT', 'COICOP'], inplace=True, axis=1)
df.info()   # time as object, need convert to data_time format
df['TIME'].replace(regex=True,inplace=True, to_replace='M',value='')

df['TIME'] =  pd.to_datetime(df['TIME'], format='%Y%m', errors='ignore', infer_datetime_format=True)

df = df.set_index(['TIME'])
df.index
df[pd.isnull(df['Value'])].count()
# Resulting Plot

rcParams['figure.figsize'] = 12, 6

df.plot()

plt.xlabel('Date')

plt.ylabel('Index')

plt.title("HICP")
# Test stationarity for model selection



from statsmodels.tsa.stattools import adfuller

def testStationarity(ts):

    dftest = adfuller(ts)

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    return dfoutput



testStationarity(df.Value)
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df, model='multiplicative')





rcParams['figure.figsize'] = 12, 6

plt.rc('lines', linewidth=1, color='r')



fig = result.plot()

#print(plt.style.available)
import statsmodels.api as sm

mod = sm.tsa.statespace.SARIMAX(df,

                                order=(1, 1, 0),

                                seasonal_order=(0, 1, 1, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)



results = mod.fit()



print(results.summary())
results.plot_diagnostics(figsize=(14,10))

plt.show()
pred = results.get_prediction(start=pd.to_datetime('2016-01-01'), dynamic=False)

pred_ci = pred.conf_int()
pred_ci['Predicted'] = (pred_ci['lower Value'] + pred_ci['upper Value'])/2

pred_ci['Observed'] = df['Value']

pred_ci['Diff, %%'] = ((pred_ci['Predicted'] / pred_ci['Observed'])-1) * 100

pred_ci.tail(5)
ax = df['1990':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7)



ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)



plt.xlabel('Date')

plt.ylabel('Index')

plt.title("HICP")

plt.legend()

rcParams['figure.figsize'] = 12, 10

plt.show()
# Get forecast 3 years ahead in future

pred_uc = results.get_forecast(steps=36)



# Get confidence intervals of forecasts

pred_ci = pred_uc.conf_int()
ax = df.plot(label='observed', figsize=(12, 6))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

plt.xlabel('Date')

plt.ylabel('Index')

plt.title("HICP")



plt.legend()



plt.show()
pred_ci.head(11)
rcParams['figure.figsize'] = 12, 6

pred_ci.head(11).plot()

plt.xlabel('Date')

plt.ylabel('Index')

plt.title("HICP")
pred_ci['Mean'] = (pred_ci['lower Value'] + pred_ci['upper Value'])/2
pred_ci['Mean'].head(11)
rcParams['figure.figsize'] = 12, 6

pred_ci['Mean'].head(11).plot()

plt.xlabel('Date')

plt.ylabel('Index')

plt.title("HICP")
import seaborn as sns



print("                     HICP predicted monthly and annual rates of change Jan 2019 to Dec 2019 ")

monthly_roc = [-0.8, 0.3, 0.8, 0.4, 0.3, 0.1, -0.3, 0.2, 0.3, 0.2, -0.2, 0.2]

annual_roc = [1.5, 1.6, 1.5, 1.5, 1.2, 1.2, 1.1, 1, 1, 0.9, 1.1, 1.3]

index = ['2019-01', '20190-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12']





f, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)



# Generate some sequential data



sns.barplot(x=index, y=monthly_roc, palette="rocket", ax=ax1)

ax1.axhline(0, color="k", clip_on=False)

ax1.set_ylabel("HICP monthly rate of change")



# Center the data to make it diverging



sns.barplot(x=index, y=annual_roc, palette="vlag", ax=ax2)

ax2.axhline(0, color="k", clip_on=False)

ax2.set_ylabel("HICP annual rate of change")





print("                     HICP original and predicted monthly and annual rates of change Jan 2018 to Dec 2019 ")



monthly_rocc = [-0.7, 0.2, 0.8, 0.4, 0.5, 0.1, -0.2, 0.2, 0.3, 0.2, -0.4, 0.0,

-0.8, 0.3, 0.8, 0.4, 0.3, 0.1, -0.3, 0.2, 0.3, 0.2, -0.2, 0.2] 

annual_rocc = [1.6, 1.4, 1.6, 1.5, 2.0, 2.1, 2.2, 2.2, 2.2, 2.3, 2.0, 1.6,

1.5, 1.6, 1.5, 1.5, 1.2, 1.2, 1.1, 1.0, 1.0, 0.9, 1.1, 1.3]

index = ['201801', '201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812', '201901', '2019002', '201903', '201904', '201905', '201906', '201907', '201908', '201909', '201910', '201911', '201912']





f, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)



# Generate some sequential data



sns.barplot(x=index, y=monthly_rocc, palette="rocket", ax=ax1)

ax1.axhline(0, color="k", clip_on=False)

ax1.set_ylabel("HICP monthly rate of change")



# Center the data to make it diverging



sns.barplot(x=index, y=annual_rocc, palette="vlag", ax=ax2)

ax2.axhline(0, color="k", clip_on=False)

ax2.set_ylabel("HICP annual rate of change")
