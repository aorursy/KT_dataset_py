import pandas as pd

import numpy as np

pd.set_option("display.precision", 3)

pd.set_option("display.expand_frame_repr", False)

pd.set_option("display.max_rows", 25)

import matplotlib.pyplot as plt

#COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

#COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

#time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

#time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

#time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

#print(covid_19_data.columns)

#print('Countries:',countries)
by_country = covid_19_data.groupby(['Country/Region','ObservationDate'])[['Confirmed','Recovered','Deaths']].agg("sum")

#by_country.loc[['Australia','Italy','South Korea']]

# this creates a multi-index. flatten it

flat = by_country.reset_index()

countries = set(list(flat['Country/Region']))

#flat.loc[flat['Country/Region']=='occupied Palestinian territory']

by_date = covid_19_data.groupby(['ObservationDate','Country/Region'])[['Confirmed']].agg("sum")

# To get countries as columns

country_cols = by_date['Confirmed'].unstack()

country_cols[['Australia','Italy','South Korea','Singapore']].plot()

#country_cols[['Australia','Singapore']].plot()
val = 100

K1 = by_date.groupby(['Country/Region']).agg(

    iK = pd.NamedAgg(column='Confirmed', aggfunc=lambda x: abs(x-val).idxmin() )

)

aggs = by_date.groupby(['Country/Region']).agg(

    max_confirmed = pd.NamedAgg(column='Confirmed', aggfunc=max),

    min_confirmed = pd.NamedAgg(column='Confirmed', aggfunc=min)

)

# You can't have more than one lambda function in a set of aggregations

aggs_lamb = by_date.groupby(['Country/Region']).agg(

    mean = pd.NamedAgg(column='Confirmed', aggfunc=lambda x: np.mean(x) ),

    #maxm = pd.NamedAgg(column='Confirmed', aggfunc=lambda x: np.max(x) )

)
idx = K1.loc['Venezuela']

by_date.loc[idx]
aus = covid_19_data.loc[covid_19_data['Country/Region']=='Australia']

italy = covid_19_data.loc[covid_19_data['Country/Region']=='Italy']

print(aus.tail())

ausbydate = aus.groupby(['ObservationDate'])

italbydate = italy.groupby(['ObservationDate'])

total_aus = ausbydate.agg(np.sum)

total_ital = italbydate.agg(np.sum)
val1 = 10

val2 = 1000

idx1,idx2 = abs(total_ital['Confirmed'] - val1).idxmin(),abs(total_ital['Confirmed'] - val2).idxmin()

ital_to_1K = total_ital.loc[idx1:idx2]
ital_to_1K['Confirmed'].plot()
val1 = 50

val2 = 2000

idx1,idx2 = abs(total_aus['Confirmed'] - val1).idxmin(),abs(total_aus['Confirmed'] - val2).idxmin()

aus_to_1K = total_aus.loc[idx1:idx2]

print(aus_to_1K)
total_aus['death_frac'] = total_aus['Deaths']/total_aus['Confirmed']

print("{:4.2f}%".format(100*total_aus['death_frac'].iloc[total_aus.shape[0]-1]))
val = 500

ital_index = abs(total_ital['Confirmed'] - val).idxmin()

aus_index = abs(total_aus['Confirmed'] - val).idxmin()

print(total_ital.loc[ital_index])

print(total_aus.loc[aus_index])
#plt.clf()

##fig = plt.figure(figsize=[6,6])

#ax1 = plt.subplot(2,1,1)

#total_aus['Confirmed'].plot(ax=ax1)

#plt.title('AUS')

#plt.subplot(2,1,2)

#plt.title('ITALY')

#total_ital['Confirmed'].plot()

#plt.subplots_adjust(hspace=0.8)

#fig.autofmt_xdate()
plt.clf()

fig = plt.figure(figsize=[12,6])

ax1 = plt.subplot(1,2,1)

total_aus['Confirmed'].plot()

plt.plot(total_aus.index,total_aus['Confirmed'])

plt.plot(aus_index,total_aus.loc[aus_index]['Confirmed'],'bo')

plt.title('AUS')

plt.subplot(1,2,2)

plt.title('ITALY')

total_ital['Confirmed'].plot()

plt.plot(total_ital.index,total_ital['Confirmed'])

plt.plot(ital_index,total_ital.loc[ital_index]['Confirmed'],'bo')

plt.subplots_adjust(hspace=0.8)

fig.autofmt_xdate()