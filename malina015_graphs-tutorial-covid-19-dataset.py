import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
covid_19_data =  pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid_19_data.head()
south_korea_data = covid_19_data[covid_19_data['Country/Region']=='South Korea']

south_korea_data.head()
south_korea_data_groupped = south_korea_data.groupby(['ObservationDate','Country/Region']).agg({'Confirmed':sum,

                                                                'Deaths':sum,

                                                                'Recovered':sum})

south_korea_data_groupped.head(15)
south_korea_data_groupped = south_korea_data_groupped.reset_index()

south_korea_data_groupped.head()
fig, ax1 = plt.subplots()

ax1.bar(south_korea_data_groupped['ObservationDate'], south_korea_data_groupped['Confirmed'])

ax1.bar(south_korea_data_groupped['ObservationDate'], south_korea_data_groupped['Deaths'])

plt.show()
import datetime

south_korea_data_groupped['ObservationDate'] = south_korea_data_groupped['ObservationDate'].apply(lambda row: datetime.datetime.strptime(row,"%m/%d/%Y").date())

south_korea_data_groupped.head()
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import matplotlib.dates as mdates

fig, ax1 = plt.subplots()

formatter = mdates.DateFormatter("%Y-%m-%d")

ax1.xaxis.set_major_formatter(formatter)   

dates_list = list(south_korea_data_groupped['ObservationDate'].unique())

ax1.bar(dates_list, south_korea_data_groupped['Confirmed'])

ax1.bar(dates_list, south_korea_data_groupped['Deaths'])

ax1.tick_params(axis='y')

plt.show()
fig, ax1 = plt.subplots(figsize=(10,5))

fig.autofmt_xdate()

formatter = mdates.DateFormatter("%Y-%m-%d")

ax1.xaxis.set_major_formatter(formatter)   

dates_list = list(south_korea_data_groupped['ObservationDate'].unique())

ax1.bar(dates_list, south_korea_data_groupped['Confirmed'])

ax1.bar(dates_list, south_korea_data_groupped['Deaths'])

ax1.tick_params(axis='y')

plt.show()
fig, ax1 = plt.subplots()

ax1.set(xlabel='Dates', ylabel='Number of cases', title='South Korea')

formatter = mdates.DateFormatter("%Y-%m-%d")

ax1.xaxis.set_major_formatter(formatter)   

dates_list = list(south_korea_data_groupped['ObservationDate'].unique())

ax1.bar(dates_list, south_korea_data_groupped['Confirmed'], label='Confirmed')

ax1.bar(dates_list, south_korea_data_groupped['Deaths'], label='Deaths')

ax1.tick_params(axis='y')

plt.xticks(rotation=45)

plt.legend()

plt.show()
skdg = south_korea_data_groupped

#skdg['Deaths_yesterday'] = skdg.groupby(['Country/Region'])['Deaths'].shift(1)

skdg['Deaths_yesterday'] = skdg['Deaths'].shift(1)

skdg['Deaths_per_day'] = skdg['Deaths']- skdg['Deaths_yesterday']

#skdg['Confirmed_yesterday'] = skdg.groupby(['Country/Region'])['Confirmed'].shift(1)

skdg['Confirmed_yesterday'] = skdg['Confirmed'].shift(1)

skdg['Confirmed_per_day'] = skdg['Confirmed']- skdg['Confirmed_yesterday']

skdg['Death_rate']=skdg['Deaths']/skdg['Confirmed']

skdg['Death_rate_ended']=skdg['Deaths']/(skdg['Deaths']+skdg['Recovered'])
skdg.head()
fig, ax1 = plt.subplots()

ax1.set(xlabel='Dates', ylabel='Number of cases', title='South Korea')

formatter = mdates.DateFormatter("%Y-%m-%d")

ax1.xaxis.set_major_formatter(formatter)   

dates_list = list(skdg['ObservationDate'].unique())

ax1.bar(dates_list, skdg['Confirmed'], label='Confirmed')

ax1.bar(dates_list, skdg['Deaths'], label='Deaths')

ax1.plot(dates_list , skdg['Death_rate_ended'], label = 'Death rate closed cases')

ax1.plot(dates_list , skdg['Death_rate'], label = 'Death rate')

ax1.tick_params(axis='y')

plt.xticks(rotation=45)

plt.legend()

plt.show()
fig, ax1 = plt.subplots()

ax1.set(xlabel='Dates', ylabel='Number of cases', title='South Korea')

formatter = mdates.DateFormatter("%Y-%m-%d")

ax1.xaxis.set_major_formatter(formatter)

plt.xticks(rotation=45)

ax2 = ax1.twinx() 

dates_list = list(skdg['ObservationDate'].unique())

b1 = ax1.bar(dates_list, skdg['Confirmed'], label='Confirmed')

b2 = ax1.bar(dates_list, skdg['Deaths'], label='Deaths')

l1 = ax2.plot(dates_list , skdg['Death_rate_ended'], label = 'Death rate closed cases')

l2 = ax2.plot(dates_list , skdg['Death_rate'], label = 'Death rate')

ax1.tick_params(axis='y')

#plt.legend()

lines, labels = ax1.get_legend_handles_labels()

lines2, labels2 = ax2.get_legend_handles_labels()

ax2.legend(lines + lines2, labels + labels2)

plt.show()
from bokeh import palettes as bh

colors = bh.all_palettes['PiYG'][4]

fig, ax1 = plt.subplots()

ax1.set(xlabel='Dates', ylabel='Number of cases', title='South Korea')

formatter = mdates.DateFormatter("%Y-%m-%d")

ax1.xaxis.set_major_formatter(formatter)

plt.xticks(rotation=45)

ax2 = ax1.twinx() 

dates_list = list(skdg['ObservationDate'].unique())

b1 = ax1.bar(dates_list, skdg['Confirmed'], label='Confirmed', color = colors[1])

b2 = ax1.bar(dates_list, skdg['Deaths'], label='Deaths', color = colors[2])

l1 = ax2.plot(dates_list , skdg['Death_rate_ended'], label = 'Death rate closed cases', color = colors[0])

l2 = ax2.plot(dates_list , skdg['Death_rate'], label = 'Death rate', color = colors[3])

ax1.tick_params(axis='y')



lines, labels = ax1.get_legend_handles_labels()

lines2, labels2 = ax2.get_legend_handles_labels()

ax2.legend(lines + lines2, labels + labels2)

plt.show()
import matplotlib.ticker as mtick

plt.style.use('bmh')

fig, ax1 = plt.subplots(figsize=(10,5))

ax1.set(xlabel='Dates', ylabel='Number of cases', title='South Korea')



formatter = mdates.DateFormatter("%Y-%m-%d")

ax1.xaxis.set_major_formatter(formatter)

plt.xticks(rotation=45)

ax2 = ax1.twinx() 

ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))

dates_list = list(skdg['ObservationDate'].unique())

b1 = ax1.bar(dates_list, skdg['Confirmed'], label='Confirmed', color = colors[1])

b2 = ax1.bar(dates_list, skdg['Deaths'], label='Deaths', color = colors[2])

l1 = ax2.plot(dates_list , skdg['Death_rate_ended'], label = 'Death rate closed cases', color = colors[0])

l2 = ax2.plot(dates_list , skdg['Death_rate'], label = 'Death rate', color = colors[3])

ax1.tick_params(axis='y')



lines, labels = ax1.get_legend_handles_labels()

lines2, labels2 = ax2.get_legend_handles_labels()

ax2.legend(lines + lines2, labels + labels2,loc=2)

ax1.set_ylim(0,10000)

ax2.set_ylim(0,0.5)

plt.show()
skdg_50 = skdg[skdg['Confirmed']>=50]

skdg_50["date_int"] = skdg_50["ObservationDate"].apply(lambda row: int(row.strftime("%Y%m%d%H%M%S")))

skdg_50["day_no"] = skdg_50.groupby("Country/Region")["date_int"].rank("dense", ascending=True)
plt.style.use('seaborn-whitegrid')

p = ['#b50007','#ffb42b','#5d3baf','#35B778']

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set(xlabel='Day since patient 50th', ylabel='Death rate',

       title='South Korea') 

ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))

x_values = list(skdg_50['day_no'].astype(int).unique())

ax2 = ax1.twinx() 

ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))

ax2.plot(x_values , skdg_50['Death_rate_ended'],  color=p[1],

         linewidth=2, label = 'Death rate closed cases')

ax2.plot(x_values , skdg_50['Death_rate'],  color=p[3],

         linewidth=2, label = 'Death rate')

ax1.bar(x_values, skdg_50['Confirmed'], label='Confirmed',  color=p[2])

ax1.bar(x_values, skdg_50['Deaths'], label='Deaths',  color=p[0])

ax1.tick_params(axis='y')

ax1.set_ylim(0,10000)

ax2.set_ylim(0,0.5)

ax1.tick_params(axis='y')



ax1.legend(bbox_to_anchor=(0.16, 0.89))

ax2.legend(bbox_to_anchor=(0.27, 1))

#fig.savefig("South Korea.png")

plt.show()