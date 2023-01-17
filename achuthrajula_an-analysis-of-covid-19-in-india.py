import numpy as np

import pandas as pd

import matplotlib

from datetime import datetime

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.neural_network import MLPRegressor

from fbprophet import Prophet
data = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

data = pd.DataFrame(data)

data1 = data

total = data1['ConfirmedIndianNational'] + data1['ConfirmedForeignNational']

data1['y'] = total

data1.rename(columns = {'Date':'ds'}, inplace = True) 

data = data1.drop(['Sno', 'State/UnionTerritory'], axis = 1)

data1 = data1.drop(['Sno', 'State/UnionTerritory', 'ConfirmedIndianNational', 'ConfirmedForeignNational', 'Cured', 'Deaths'], axis = 1)

data1.head()
total_rows = np.shape(data)[0]

i = 0

dates = {}

l = []

while i != total_rows:

    if data.iloc[i,0] in dates:

        dates[str(data.iloc[i,0])]['ConfirmedIndianNational'] += data.iloc[i,1]

        dates[str(data.iloc[i,0])]['ConfirmedForeignNational'] += data.iloc[i,2]

        dates[str(data.iloc[i,0])]['Cured'] += data.iloc[i,3]

        dates[str(data.iloc[i,0])]['Deaths'] += data.iloc[i,4]

    else:

        l.append(data.iloc[i,0])

        dates[str(data.iloc[i,0])] = {

            'ConfirmedIndianNational': data.iloc[i,1],

            'ConfirmedForeignNational': data.iloc[i,2],

            'Cured': data.iloc[i,3],

            'Deaths': data.iloc[i,4]

        }

    i += 1

week_dates = [l[k:k+7] for k in range(0, len(l), 7)]

no_of_weeks = len(week_dates)

new_data = pd.DataFrame(dates)

new_data = new_data.T

total = new_data['ConfirmedIndianNational'] + new_data['ConfirmedForeignNational']

new_data = new_data.drop(['ConfirmedIndianNational', 'ConfirmedForeignNational'], axis = 1)

new_data['ConfirmedCases'] = total

new_data= new_data.T

new_data.head()

# new_data['ConfirmedCases'] = data['ConfirmedIndianNational'] + data['ConfirmedForeignNational']
new_data

s=slice(30,40)

date = []

for i in range(len(l)):

    date.append(matplotlib.dates.date2num(datetime.strptime(l[i], '%d/%m/%y')))

ax = plt.gca()

# ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator(interval=3))

# ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%e'))

ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())

ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b'))

plt.xlabel("-----| Time |---->")

plt.ylabel("-----| Total Confirmed cases |---->")

plt.plot(date,new_data.iloc[2,:])

plt.show()
m = Prophet()

m.fit(data1)

future = m.make_future_dataframe(periods=7)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
ig2 = m.plot_components(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
