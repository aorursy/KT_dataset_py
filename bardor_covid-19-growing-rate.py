# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#%% load data

import datetime

import seaborn as sns

import plotly.offline as py

import plotly.express as px

#from fbprophet import Prophet

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from sklearn.cluster import KMeans

from datetime import date, timedelta

from plotly.subplots import make_subplots

from statsmodels.tsa.arima_model import ARIMA

#from fbprophet.plot import plot_plotly, add_changepoints_to_plot



fn = '/kaggle/input/corona-virus-report/covid_19_clean_complete.csv'

data1=pd.read_csv(fn)



daily = data1.sort_values(['Date','Country/Region','Province/State'])

latest = data1[data1.Date == daily.Date.max()]

data=latest.rename(columns={ "Country/Region": "country", "Province/State": "state","Confirmed":"confirm","Deaths": "death","Recovered":"recover"})

dgc=data.groupby("country")[['confirm', 'death', 'recover']].sum().reset_index()

#%% data to array, for analysis

def data2relevantArray(active,th):

    tmp=active.array

    ok=True

    c=len(tmp)-1

    while ok:

        if tmp[c]<th:

            ok=False

        elif c == 1:

            ok = False

        c=c-1

    tmp=tmp[c:]

    return tmp

#%% Hubei, China data

coun_data = data1[data1['Country/Region']=='China']

coun_data = coun_data[coun_data['Province/State']=='Hubei']

coun_data['Date'] = pd.to_datetime(coun_data['Date'])

coun_data = coun_data.sort_values(by='Date')

active = coun_data['Confirmed']-coun_data['Recovered']-coun_data['Deaths']

f, ax = plt.subplots(figsize=(17,10))

plt.plot(coun_data.Date,coun_data.Confirmed,zorder=1,color="black")

plt.plot(coun_data.Date,coun_data.Recovered,zorder=1,color="green")

plt.plot(coun_data.Date,coun_data.Deaths,zorder=1,color="red")

plt.plot(coun_data.Date,active,zorder=1,color="magenta",label = "Active")

plt.title('China -- Hubei',fontsize=16)

ax.set_xlim([datetime.date(2020, 1, 22), coun_data['Date'].iloc[-1]])

ax.set_ylim([-1, max(coun_data['Confirmed']) + 20])

plt.legend()

f.autofmt_xdate()

active0 = active

#%% Italy, Israel and some countries data

countraies = ['Italy','Israel','Japan','Iran','South Korea','Spain','Taiwan*','US']

for countray in countraies:

    coun_data = data1[data1['Country/Region']==countray]

    coun_data['Date'] = pd.to_datetime(coun_data['Date'])

    count_data = coun_data.sort_values(by='Date')

    active = coun_data['Confirmed']-coun_data['Recovered']-coun_data['Deaths']

    f, ax = plt.subplots(figsize=(17,10))

    plt.plot(coun_data.Date,coun_data.Confirmed,zorder=1,color="black")

    plt.plot(coun_data.Date,coun_data.Recovered,zorder=1,color="green")

    plt.plot(coun_data.Date,coun_data.Deaths,zorder=1,color="red")

    plt.plot(coun_data.Date,active,zorder=1,color="magenta",label = "Active")

    plt.title(countray + ' - ' + str(coun_data['Date'].iloc[-1]),fontsize=16)

    ax.set_xlim([datetime.date(2020, 2, 22), coun_data['Date'].iloc[-1]])

    ax.set_ylim([-1, max(coun_data['Confirmed']) + 20])

    plt.legend()

    f.autofmt_xdate()

#%% active cases for the above countries

f, ax = plt.subplots(figsize=(17,10))

plt.plot(coun_data.Date,active0,zorder=1,label = 'Hubei, China')

m=list([max(active0)])

for countray in countraies:

    coun_data = data1[data1['Country/Region']==countray]

    coun_data['Date'] = pd.to_datetime(coun_data['Date'])

    coun_data = coun_data.sort_values(by='Date')

    active[countray] = coun_data['Confirmed']-coun_data['Recovered']-coun_data['Deaths']

    plt.plot(coun_data.Date,active[countray],zorder=1,label = countray)

    m.append(active[countray].max())

ax.set_xlim([datetime.date(2020, 1, 22), coun_data['Date'].iloc[-1]])

ax.set_yscale("log", nonposy='clip')

ax.set_ylim([0.9, max(m) + 2000])

plt.legend(fontsize=16, loc='upper left')

f.autofmt_xdate()

plt.grid()

plt.title('Corona virus situation' + ' - ' + str(coun_data['Date'].iloc[-1]),fontsize=16)

plt.ylabel('Active Cases',fontsize=16)

#%% Active Cases since the 20th active case

th=20

f, ax = plt.subplots(figsize=(14,10))

tmp = data2relevantArray(active0,th)

m=list([max(active0)])

for countray in countraies:

    coun_data = data1[data1['Country/Region']==countray]

    coun_data['Date'] = pd.to_datetime(coun_data['Date'])

    coun_data = coun_data.sort_values(by='Date')

    active2 = coun_data['Confirmed']-coun_data['Recovered']-coun_data['Deaths']

    tmp = data2relevantArray(active2,th)

    plt.plot(range(len(tmp)),np.array(tmp),zorder=1,linewidth=3,label = countray)

    plt.plot(len(tmp)-1,np.array(tmp[-1]),'ko',zorder=1)

    plt.text(len(tmp)-0.5,np.array(tmp[-1]),countray)

    m.append(active[countray].max())

ax.set_xlim([0.9,50])

ax.set_yscale("log", nonposy='clip')

ax.set_ylim([th-1, max(m) + 20000])

plt.plot([1,34],[th,th*1.25**33],'k--',linewidth=2,label = 'Growing rate = 25% a day')

plt.text(34,th*1.25**33,'Double the numbers every three days')

plt.plot([12,35],[150,150*1.08**(34-12)],'g:',linewidth=2,label = 'Growing rate = 8% a day')

plt.text(15,270,'Growing rate = 8% a day',rotation=17)

plt.plot([8,33],[60,60*1.36**(33-8-1)],'k:',linewidth=2,label = 'Growing rate = 36% a day')

plt.text(14,5000,'Double the numbers every 2.25 days',rotation=47)

plt.plot([1,6],[th,th*3**5],'k-.',linewidth=2,label = 'Growing rate = 200% a day')

plt.text(4,th*3**6,'Triple the numbers every day',rotation=77)

f.autofmt_xdate()

plt.grid()

plt.title('Corona virus situation' + ' - ' + str(coun_data['Date'].iloc[-1]),fontsize=16)

plt.ylabel('Active Cases',fontsize=16)

plt.xlabel('Days since the '+str(th)+'th active case',fontsize=16)

plt.text(48,22,'D.B.')
