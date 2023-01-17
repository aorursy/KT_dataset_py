# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from pathlib import Path

data_dir = Path('/kaggle/input/covid19-in-india')



os.listdir(data_dir)
data = pd.read_csv(data_dir/'covid_19_india.csv')

data.head()
data.Date=pd.to_datetime(data.Date,dayfirst=True)
print(f"Earliest Entry: {data['Date'].min()}")

print(f"Last Entry:     {data['Date'].max()}")

print(f"Total Days:     {data['Date'].max() - data['Date'].min()}")
data['confirmed']=data.ConfirmedForeignNational+data.ConfirmedIndianNational
grouped = data.groupby('Date')['Date', 'confirmed', 'Deaths'].sum().reset_index()



fig = px.line(grouped, x="Date", y="confirmed",

              title="Indiawide Confirmed Cases Over Time")

fig.show()



fig = px.line(grouped, x="Date", y="confirmed", 

              title="Indiawide Confirmed Cases (Logarithmic Scale) Over Time", 

              log_y=True)

fig.show()
data=data.rename(columns={'Date':'date',

                     'State/UnionTerritory':'state',

                         'Deaths':'deaths'})
latest = data[data['date'] == max(data['date'])].reset_index()

latest_grouped = latest.groupby('state')['confirmed', 'deaths'].sum().reset_index()
latest = data[data['date'] == max(data['date'])]

latest = latest.groupby('state')['confirmed', 'deaths'].max().reset_index()



fig = px.bar(latest.sort_values('confirmed', ascending=False)[:15][::-1], 

             x='confirmed', y='state', color_discrete_sequence=['#D63230'],

             title='Confirmed Cases in India', text='confirmed', orientation='h')

fig.show()
for s in latest.sort_values(by='confirmed',ascending = False)[:15].state:

    grouped = data[data.state==s].groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()



    fig = px.line(grouped, x="date", y="confirmed",

                  title= s+"wide Confirmed Cases Over Time")

    fig.show()



    fig = px.line(grouped, x="date", y="confirmed", 

                  title=s+"wide Confirmed Cases (Logarithmic Scale) Over Time", 

                  log_y=True)

    fig.show()
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries,state):

    

    #Determing rolling statistics

    rolmean = timeseries.resample("1d").sum().fillna(0).rolling(window=7, min_periods=1).mean()

    rolstd = timeseries.resample("1d").sum().fillna(0).rolling(window=7, min_periods=1).std()



#Plot rolling statistics:

    plt.xticks(rotation=70)

    plt.plot(timeseries, color='blue',label='Original')

    plt.xticks(rotation=70)

    plt.plot(rolmean, color='red', label='Rolling Mean')

    plt.xticks(rotation=70)

    plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation for '+ state)

    plt.show()

    #Perform Dickey-Fuller test:

    print ('Results of Dickey-Fuller Test:')

    if timeseries.shape[0] > 15:

        dftest = adfuller(timeseries, autolag='AIC')

        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

        for key,value in dftest[4].items():

            dfoutput['Critical Value (%s)'%key] = value

        print (dfoutput)

    else:

        print ('Insufficient Data')
for s in latest.sort_values(by='confirmed',ascending = False)[:10].state:

    grouped = data[data.state==s].groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

    g=grouped.set_index('date')[['confirmed']]

    test_stationarity(g,s)

    