import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import matplotlib.ticker as ticker

import seaborn as sns

import datetime

import matplotlib.ticker as ticker



import warnings

warnings.filterwarnings("ignore")

df_train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")
country = df_train['Country/Region'].unique()

country
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_train =df_train.drop(['Id', 'Lat', 'Long', 'Province/State'], axis=1)

df_train
#Focusing on Japan．



#identifing the country.

id_x = np.where(country == 'Japan')

id_x = id_x[0][0]

df_a = df_train[df_train['Country/Region'].isin([country[id_x]])]

df_a["ConfirmedCases"]  = df_a["ConfirmedCases"].diff(1)

df_a[country[id_x]+'_7-dayAverage'] =df_a[df_train.columns[2]].rolling(7).mean().round(1)

df_a = df_a.dropna(how = 'any') 





#Visualization.

fig, ax  = plt.subplots(figsize=(5, 5))



ax.bar(x = df_a['Date'], height = df_a['ConfirmedCases'], color = 'mistyrose', label = "New cases")

ax.plot(df_a['Date'], df_a[country[id_x]+'_7-dayAverage'], color = 'red',label = "7-day average")

ax.set_xlabel("Date")

ax.set_ylabel("Confirmed cases")

plt.rcParams["font.size"] = 10

ax.xaxis.set_major_locator(ticker.MultipleLocator(30.00))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.title(str(country[id_x]),fontweight="bold")

plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=1, fontsize=10)

plt.show()
#Visualization in the world.



fig, ax  = plt.subplots(dpi=100, figsize=(60, 120))

plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.gca().spines['right'].set_visible(False)

plt.gca().spines['top'].set_visible(False)

plt.gca().spines['left'].set_visible(False)

plt.gca().spines['bottom'].set_visible(False)

plt.tick_params(labelbottom=False)

plt.tick_params(bottom=False)



for i in  range(len(country)):

    ax = fig.add_subplot(17, 10, i+1)

    df_a = df_train[df_train['Country/Region'].isin([country[i]])]

    df_a["ConfirmedCases"]  = df_a["ConfirmedCases"].diff(1)

    df_a[country[0]+'_7-dayAverage'] =df_a[df_train.columns[2]].rolling(7).mean().round(1)

    df_a = df_a.dropna(how = 'any') 



    ax.bar(x = df_a['Date'], height = df_a['ConfirmedCases'], color = 'mistyrose', label = "New cases")

    ax.plot(df_a['Date'], df_a[country[0]+'_7-dayAverage'], color = 'red',label = "7-day average")



    ax.set_xlabel("Date")

    ax.set_ylabel("Confirmed cases")

    plt.rcParams["font.size"] = 10

    ax.xaxis.set_major_locator(ticker.MultipleLocator(30.00))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.title(str(country[i]),fontweight="bold")

    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=1, fontsize = 7)



    

#general title

plt.suptitle("Where Countries Are on the Curve", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)



dt_today = datetime.date.today()

#plt.savefig(str(dt_today) + "_COVID-19_timeseries.png") 