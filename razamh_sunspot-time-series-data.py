# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

mpl.rcParams['figure.figsize'] = (15, 7)

mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (20, 7)

mpl.rcParams['axes.grid'] = False
df = pd.read_csv("/kaggle/input/sunspots/Sunspots.csv")

df.head()
df.info()
from dateutil.parser import parse

dateparse=lambda dates:parse(dates)
df = pd.read_csv('/kaggle/input/sunspots/Sunspots.csv',usecols=['Date','Monthly Mean Total Sunspot Number'],parse_dates=['Date'],date_parser=dateparse)

df.head()
df.info() ## Checking the info again : data type of Date column --> has conveted into datetime
df_non_index=df.copy() # Making a copy of initial data.Both will be used as required

# The 'df_non_index' dataframe is used for some exploratory data analysis  

# Later we will convert Date colum as index in  'df' dataframe.
df_non_index['Month']=df_non_index.Date.dt.month

df_non_index.head()
df_non_index['nth_year'] =[int(str(i)[3]) for i in (df_non_index.Date.dt.year)] # Note this is list comprehension 

df_non_index['nth_year'].replace(0,10,inplace=True)

df_non_index.head(10)
fig, axes = plt.subplots(3, 1, figsize=(20,15), dpi= 80)

sns.boxplot(x='Date', y='Monthly Mean Total Sunspot Number', data=df_non_index, ax=axes[0])

sns.boxplot(x='Month', y='Monthly Mean Total Sunspot Number', data=df_non_index,ax = axes[1])

sns.boxplot(x='nth_year', y='Monthly Mean Total Sunspot Number', data=df_non_index,ax = axes[2])

# Set Title

axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=14); 

axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=14)

axes[2].set_title('nth_year_each_decade\n(The Seasonality)', fontsize=14)

fig.tight_layout()

plt.show()
df = df.set_index('Date')

df.head()
df.tail()
df.plot(grid=True) # plots in pandas itself take index as x axis, here it is datetime and y axis is  'Monthly Mean Total Sunspot Number'

#  This plot is same to that of previous first box plot (that was a scatter plot, here it dots are joined )
df_2018=df.loc['2000':'2010'] # Slicing all data from 2000 to 2010

df_2018.plot(figsize=(16,7),grid=True)
df_2018=df.loc['1900':'1920'] # Slicing all data from 1900 to 1910

df_2018.plot(figsize=(16,7),grid=True)

plt.show()
import plotly.express as px  

fig = px.line(df_non_index, x='Date', y='Monthly Mean Total Sunspot Number', title='Mean_Sunspot_Slider')

fig.update_xaxes(rangeslider_visible=False)

fig.show()

## There is slider belwo the graph using which we can select any particular time zone
fig = px.line(df_non_index, x='Date', y='Monthly Mean Total Sunspot Number', title='Mean_Sunspot_Slider')



fig.update_xaxes(

    rangeslider_visible=False,

    rangeselector=dict(

        buttons=list([

            dict(count=10, label="10y", step="year", stepmode="backward"),

            dict(count=20, label="20y", step="year", stepmode="backward"),

            dict(count=30, label="30y", step="year", stepmode="backward"),

            dict(count=40, label="40y", step="year", stepmode="backward"),

            dict(count=50, label="50y", step="year", stepmode="backward"),

            dict(step="all")

        ])

    )

)

fig.show()
df_non_index.head()
df_11_1985=df_non_index[(df_non_index.Date.dt.year>=1985) & (df_non_index.Date.dt.year<1996)]

df_11_1996=df_non_index[(df_non_index.Date.dt.year>=1996) &(df_non_index.Date.dt.year<2007)]



x=np.arange(1,len(df_11_1996['Date'])+1)



plt.plot(x, df_11_1985['Monthly Mean Total Sunspot Number'],label='df_60_1998')

plt.plot(x, df_11_1996['Monthly Mean Total Sunspot Number'],label='df_60_1958')

plt.legend()

plt.xlabel('Month')

plt.ylabel('Monthly Mean Total Sunspot Number')

plt.title('Comparison of Two consecutive 11 year')

plt.show()
fig=plt.figure(figsize=(18,6))

fig.subplots_adjust(hspace=0.4, wspace=0.2)

ax1=fig.add_subplot(2,2,1)

pd.plotting.lag_plot(df['Monthly Mean Total Sunspot Number'],lag=1)

plt.title('Lag_1')

ax2=fig.add_subplot(2,2,2)

pd.plotting.lag_plot(df['Monthly Mean Total Sunspot Number'],lag=3)

plt.title('Lag_3')

ax3=fig.add_subplot(2,2,3)

pd.plotting.lag_plot(df['Monthly Mean Total Sunspot Number'],lag=6)

plt.title('Lag_6')

ax3=fig.add_subplot(2,2,4)

pd.plotting.lag_plot(df['Monthly Mean Total Sunspot Number'],lag=24)

plt.title('Lag_24')

plt.show()
fig=plt.figure(figsize=(18,6))

fig.subplots_adjust(hspace=0.4, wspace=0.2)

ax1=fig.add_subplot(1,2,1)

df['Monthly Mean Total Sunspot Number'].hist()

plt.title('Histogram')

ax2=fig.add_subplot(1,2,2)

df['Monthly Mean Total Sunspot Number'].plot(kind='density')# kernel density plot

plt.title('KDE')

plt.show()
from statsmodels.tsa.stattools import adfuller
data_series=df['Monthly Mean Total Sunspot Number']
print('Results of Dickey-Fuller Test:')

dftest = adfuller(data_series, autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print(dfoutput)

if dfoutput['Test Statistic'] < dfoutput['Critical Value (5%)']:  ## Comparing with 5% significant Level

  print('Series is stationary')

else:

  print('Series is not Stationary')

## OR 

if dfoutput[1] > 0.05 :

  print('Series is not Stationary')

else:

  print('Series is Stationary')
from statsmodels.tsa.stattools import kpss
stats, p, lags, critical_values = kpss(df['Monthly Mean Total Sunspot Number'], 'c',nlags='legacy')

## pass --> 'ct' if there is trend component in data 

## pass --> 'c' if there is no trend component in data. In this case there is not trend in the data being stationary data.
print(f'Test Statistics: {stats}')

print(f'p-value: {p}')

print(f'Critial Values: {critical_values}')



if p < 0.05 :

  print('Series is not Stationary')

else:

  print('Series is Stationary')
df['Monthly Mean Total Sunspot Number'][:200].plot() # Checking for only first 200 data set

df['Monthly Mean Total Sunspot Number'][:200].rolling(3).mean().plot(label='rolling mean') ## rolling average with 3 time step also known as window

#df['Monthly Mean Total Sunspot Number'][:200].rolling(3).std().plot(label='rolling std')

plt.legend()

plt.title('Rolling Mean & Standard Deviation')

## df['Monthly Mean Total Sunspot Number'].rolling(12).mean().shift(1) # Rolling mean with shift

plt.show()
## Making a function for calculating weighted average which is passed through .apply()

def wma(weights): 

    def calc(x):

        return (weights*x).mean()

    return calc
df['Monthly Mean Total Sunspot Number'][:200].plot() # Checking for only first 200 data set

df['Monthly Mean Total Sunspot Number'][:200].rolling(3).apply(wma(np.array([0.5,1,1.5]))).plot(label='weighted mooving_averate')

#  Here inside wma 3 weights are passed since we are taking 3 time step only as window.

plt.legend()

plt.show()
df['Monthly Mean Total Sunspot Number'][:200].plot() # Checking for only first 200 data set

df['Monthly Mean Total Sunspot Number'][:200].ewm(span=3, adjust=False, min_periods=3).mean().plot(label='Exponential Weighted Average')

## Here span=3 is provide thus α=2/(span+1) automatically calculated and applied

## https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html

plt.title('Exponential Weighted M.A.')

plt.legend()

plt.show()
df['Monthly Mean Total Sunspot Number'][:200].plot() # Checking for only first 200 data set

df['Monthly Mean Total Sunspot Number'][:200].ewm(alpha=0.7, adjust=False, min_periods=3).mean().plot(label='Exponential Smooting M A')

plt.show()
df_with_diff_avg=df[:200].copy()

df_with_diff_avg['Rolling mean']=df['Monthly Mean Total Sunspot Number'][:200].rolling(3).mean()

df_with_diff_avg['W_M_A']= df['Monthly Mean Total Sunspot Number'][:200].rolling(window=3).apply(wma(np.array([0.5,1,1.5])))

df_with_diff_avg['E_W_A']= df['Monthly Mean Total Sunspot Number'][:200].ewm(span=3, adjust=False, min_periods=0).mean()

df_with_diff_avg['E_S_M_A']= df['Monthly Mean Total Sunspot Number'][:200].ewm(alpha=0.7, adjust=False, min_periods=3).mean()

print(df_with_diff_avg.head())

#df_with_diff_avg.set_index('Date', inplace=True)

df_with_diff_avg.plot()

plt.show()
df_with_diff_avg.dropna(inplace=True)
df_with_diff_avg.head()
def RMSE_CAL(df):

      Rolling_Mean_RMSE=np.sqrt(np.sum((df.iloc[:,0]-df.iloc[:,1])**2))

      W_M_A_RMSE=np.sqrt(np.sum((df.iloc[:,0]-df.iloc[:,2])**2))

      E_W_A_RMSE=np.sqrt(np.sum((df.iloc[:,0]-df.iloc[:,3])**2))

      E_S_M_A_RMSE=np.sqrt(np.sum((df.iloc[:,0]-df.iloc[:,4])**2))

      return {"Rolling_Mean_RMSE":Rolling_Mean_RMSE,"W_M_A_RMSE":W_M_A_RMSE,"E_W_A_RMSE":E_W_A_RMSE,"E_S_M_A_RMSE":E_S_M_A_RMSE}

RMSE_CAL(df_with_diff_avg)
# Additive decomposition

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['Monthly Mean Total Sunspot Number'], model="additive",freq=11*12) # Data Trend is repeated after every 11 year,freq=11*12

result.plot()

plt.show()
total_sum=result.trend+result.seasonal+result.resid

total_sum[:100] # compare this result with original Sunspot data 
df['Monthly Mean Total Sunspot Number'][:100]
pd.DataFrame(result.observed-result.trend).plot()

plt.show()
pd.plotting.autocorrelation_plot(df['Monthly Mean Total Sunspot Number']) ## for each month

plt.show()
df['Monthly Mean Total Sunspot Number'].resample("1y").mean() ## Resample based on 1 year
pd.plotting.autocorrelation_plot(df['Monthly Mean Total Sunspot Number'].resample("1y").mean())

plt.show()
from statsmodels.tsa.stattools import acf, pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Draw Plot

plot_acf(df['Monthly Mean Total Sunspot Number'].tolist(), lags=20, ax=axes[0])

plot_pacf(df['Monthly Mean Total Sunspot Number'].tolist(), lags=20, ax=axes[1])
# !pip install pmdarima
import pmdarima as pm

from pmdarima.model_selection import train_test_split



model = pm.auto_arima(df['Monthly Mean Total Sunspot Number'], 

                        m=11, seasonal=True,

                      start_p=0, start_q=0, max_order=4, test='adf',error_action='ignore',  

                           suppress_warnings=True,

                      stepwise=True, trace=True) 

 ## actually we have to set m=11*12,but it take too much time and it doesnt matter much. Source Given below:

 # https://robjhyndman.com/hyndsight/longseasonality/

model.summary()
df.reset_index(inplace=True)
df.head()
train=df[(df.Date.dt.year<1958)]

test=df[(df.Date.dt.year>=1958)]
(df.Date.dt.year>=1958) & (df.Date.dt.year<1968)
test1=df[(df.Date.dt.year>=1958) & (df.Date.dt.year<1968)]

n=len(test1)
model.fit(train['Monthly Mean Total Sunspot Number'])

forecast=model.predict(n_periods=n, return_conf_int=True)
forecast_df = pd.DataFrame(forecast[0],index = test1.index,columns=['Prediction'])
pd.concat([df['Monthly Mean Total Sunspot Number'],forecast_df],axis=1).plot()