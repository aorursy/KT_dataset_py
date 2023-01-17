# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('https://www.kaggle.com/c/covid19-global-forecasting-week-4/data'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import scipy as sp

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

import warnings 

warnings.filterwarnings('ignore')

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams



from datetime import datetime
df = pd.DataFrame([11,184], index=['Non-affected Countries', 'Affected Countries'])

df.plot(kind='pie', subplots=True,colors=['skyblue','pink'], figsize=(7, 8))
train=pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

df3 = train.groupby('Country_Region')[['ConfirmedCases', 'Fatalities']].sum().reset_index()

countries = df3['Country_Region'].unique()

print(f'{len(countries)} countries are in dataset:\n{countries}')
dfc = train.groupby(['Date', 'Country_Region'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()

today = dfc['Date'].max()

print('Date: ', today)

for i in [1, 10, 100, 1000, 10000]:

    n_countries = len(dfc.query('(Date == @today) & ConfirmedCases > @i'))

    print(f'{n_countries} countries have more than {i} ConfirmedCases')
top_countries = train.groupby('Country_Region')[['ConfirmedCases']].sum().reset_index()

less_than_100 = len(top_countries.query('(ConfirmedCases<=100)'))

cases_100to1000 = len(top_countries.query('(ConfirmedCases>=100) & (ConfirmedCases <=1000)'))

cases_1000to5000=len(top_countries.query('(ConfirmedCases>=1000) & (ConfirmedCases <=5000)'))

cases_5000to10000=len(top_countries.query('(ConfirmedCases>=5000) & (ConfirmedCases <=10000)'))

cases_10000to50000=len(top_countries.query('(ConfirmedCases>=10000) & (ConfirmedCases <=50000)'))

above_50000=len(top_countries.query('(ConfirmedCases>=50000)'))

names='less_than_100', 'cases_100to1000', 'cases_1000to5000', 'cases_5000to10000','cases_10000to50000','above_50000'

size=[16,48,31,23,37,29]

my_circle=plt.Circle( (0,0), 0.5, color='white')



plt.pie(size, labels=names, colors=['orange','purple','blue','skyblue','pink','yellow'])

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
dfc = train.groupby(['Date', 'Country_Region'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()

all_country_df = dfc.query('Date == @today')

all_country_df['casedensity'] = np.log10(all_country_df['ConfirmedCases'] + 1)

fig = px.choropleth(all_country_df, locations="Country_Region", 

                    locationmode='country names', color="casedensity", 

                    hover_name="Country_Region", hover_data=["ConfirmedCases"],

                    range_color=[all_country_df['casedensity'].min(), all_country_df['casedensity'].max()], 

                    color_continuous_scale="purples", 

                    title=f'Confirmed Case Density {today}')



trace1 = list(fig.select_traces())[0]

trace1.colorbar = go.choropleth.ColorBar(

    tickvals=[0, 1, 2, 3, 4, 5],

    ticktext=['1', '10', '100', '1000','10000'])

fig.show()
top_country_df = dfc.query('(Date == @today) & (ConfirmedCases > 1000)').sort_values('ConfirmedCases', ascending=False)



top_country_melt_df = pd.melt(top_country_df, id_vars='Country_Region', value_vars='ConfirmedCases')



fig = px.bar(top_country_melt_df, x="Country_Region", y="value", color='variable',

              title=f'Confirmed Cases as on {today}')

fig.show()
top10_countries = top_countries.sort_values('ConfirmedCases', ascending=False).iloc[:10]['Country_Region'].unique()

top10_countries_df = dfc[dfc['Country_Region'].isin(top10_countries)]

fig = px.line(top10_countries_df,

              x='Date', y='ConfirmedCases', color='Country_Region',

              title=f'Confirmed Cases for top 10 country as on {today}')

fig.show()
data=pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

data.head()
data1= data.groupby(['Date','Country_Region'])[['ConfirmedCases','Fatalities']].sum().reset_index()

data2= data1[data1['Country_Region']=='India']

data2.drop(data2[data2['ConfirmedCases']==0].index,inplace=True)

data2['Newcases']=data2['ConfirmedCases']-data2['ConfirmedCases'].shift(1)

data2['New']= data2['Newcases'].rolling(window=3).mean()

data2.drop(data2[data2['Date']<='2020-03-01'].index,inplace=True)

train1=data2.drop(['Country_Region','ConfirmedCases','Fatalities','Newcases'], axis = 1)

train1.head()
train1['Date']= pd.to_datetime(train1['Date'])

train2=train1.set_index('Date')

plt.xlabel('Date')

plt.ylabel('NewCases')

plt.plot(train2['New'])
train2.dropna(inplace=True)

rolmean = train2['New'].rolling(window=5).mean()

rolstd = train2['New'].rolling(window =5).std()
from statsmodels.tsa.stattools import adfuller

print('Dicky fuller taste')

dftest = adfuller(train2['New'],autolag='AIC')

dfout = pd.Series(dftest[0:4],index=['Test statistics','P-value','#Lags used','Number of observations'])

for key,values in dftest[4].items():

    dfout['Critical values(%s)'%key]=values

print(dfout)
train2_logscale = np.log(train2['New'])

train2_logscale.dropna(inplace=True)

plt.plot(train2_logscale)
movingAverage = train2_logscale.rolling(window=5).mean()

movingSTD = train2_logscale.rolling(window =5).std()

plt.plot(train2_logscale)

plt.plot(movingAverage,color='red')
tm_log_avg = train2_logscale-movingAverage

tm_log_avg.dropna(inplace=True)

def test_stationary(timeseries):

    movingAverage =timeseries.rolling(window=5).mean()

    movingSTD = timeseries.rolling(window=5).std()

    orig = plt.plot(timeseries,color='blue',label='Orginal')

    avg = plt.plot(movingAverage,color='black',label='Moving Average')

    std = plt.plot(movingSTD,color='red',label='Rollong std')

    plt.legend(loc='best')

    plt.title('Rolling mean and rolling std')

    plt.show()

    

    print('Dicky fuller taste')

    dftest = adfuller(train2['New'],autolag='AIC')

    dfout = pd.Series(dftest[0:4],index=['Test statistics','P-value','#Lags used','Number of observations'])

    for key,values in dftest[4].items():

        dfout['Critical values(%s)'%key]=values

        print(dfout)
test_stationary(tm_log_avg)
exponential= train2_logscale.ewm(halflife=1,min_periods=0,adjust=True).mean()

plt.plot(train2_logscale)

plt.plot(exponential, color='red')
mexponential = train2_logscale-exponential

test_stationary(mexponential)
datashifting = train2_logscale-train2_logscale.shift(1)

plt.plot(datashifting)
datashifting.dropna(inplace=True)

test_stationary(datashifting)
train2_logscale.dropna(inplace=True)

from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.api as sm



decompose = seasonal_decompose(train2_logscale)



trend = decompose.trend

seasonal=decompose.seasonal

residual=decompose.resid



plt.subplot(411)

plt.plot(train2_logscale,label='original')

plt.legend(loc='best')



plt.subplot(412)

plt.plot(trend,label='trend')

plt.legend(loc='best')



plt.subplot(413)

plt.plot(seasonal,label='seasonal')

plt.legend(loc='best')

plt.subplot(414)

plt.plot(residual,label='residual')

plt.legend(loc='best')

plt.tight_layout()



decomposelog= residual

decomposelog.dropna(inplace=True)

test_stationary(decomposelog)
decomposedlogdata = residual

decomposedlogdata.dropna(inplace=True)

test_stationary(decomposedlogdata)
from statsmodels.tsa.arima_model import ARIMA

model =ARIMA(train2_logscale,order=(2,1,0))

results_AR=model.fit(disp=-1)

plt.plot(datashifting)

plt.plot(results_AR.fittedvalues,color='red')

plt.title('RSS:%4F'%sum(results_AR.fittedvalues - datashifting**2))

print('Plotting AR model')
model =ARIMA(train2_logscale,order=(2,2,1))

results_MA =model.fit(disp=-1)

plt.plot(datashifting)

plt.plot(results_MA.fittedvalues,color='red')

print('Plotting MR model') 
model = ARIMA(train2_logscale,order=(2,0,1))

results_ARIMA= model.fit(disp=-1)

plt.plot(datashifting)

plt.plot(results_ARIMA.fittedvalues,color='red')
predictions_ARIMA_diff= pd.Series(results_ARIMA.fittedvalues,copy=True)

print(predictions_ARIMA_diff.head())
prediction_ARIMA_log=pd.Series(train2_logscale.ix[0],index= train2_logscale.index)

prediction_ARIMA_log=prediction_ARIMA_log.add(predictions_ARIMA_diff,fill_value=0)

prediction_ARIMA_log.head()
sns.set(rc={'figure.figsize':(16, 8)})

predictions_ARIMA = np.exp(prediction_ARIMA_log)

plt.plot(train2)

plt.plot(predictions_ARIMA)
results_ARIMA.plot_predict(1,70)

x=results_ARIMA.forecast(steps=70)
values0=x[0][::-1]

values0=np.exp(values0)

values1=x[1]

values1=np.exp(values1)

values= np.concatenate((values1,values0)) 

values
ts = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

ts= ts[ts['Country_Region']=='India']

ts.drop(ts[ts['ConfirmedCases']==0].index,inplace=True)

ts['New']=ts['ConfirmedCases']-ts['ConfirmedCases'].shift(1)

ts.drop(ts[ts['Date']<= '2020-03-05'].index,inplace=True)

ts1= ts.groupby('Date')[['New']].sum().reset_index()

ts2 = np.array(ts1['New'])
sns.distplot(values,hist=False,kde= True,kde_kws = {'shade': True, 'linewidth': 3})

sns.distplot(ts2, hist= False,kde=True,kde_kws = {'shade': True, 'linewidth': 3})

plt.title('Prediction')

plt.xlabel('Days')

plt.ylabel('Range')