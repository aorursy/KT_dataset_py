import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.dates import datestr2num

from datetime import datetime



plt.style.use('fivethirtyeight')



covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")



def time_series(df, name, filter=None):

    if filter:

        df = df.loc[filter(df)]

    date_list = list(df)

    date_list.remove('Lat')

    date_list.remove('Long')

    return df.loc[:, '1/22/20':].sum(axis=0).rename(name)



def show_data(df, title=None):    

    ax = df.plot()

    plt.title(title)

    plt.xlabel('Date')

    plt.ylabel('Number of cases')

    last_entry = df.iloc[-1]

    final_x = datestr2num(last_entry.name) - datestr2num(df.iloc[0].name)

    last_entries = [last_entry, df.iloc[-2], df.iloc[-3]]

    dy_latest = last_entry['Confirmed cases'] - last_entries[1]['Confirmed cases']

    dy_previous = last_entries[1]['Confirmed cases'] - last_entries[2]['Confirmed cases']

    growth_factor = dy_latest / dy_previous

    ax.annotate(f"Growth factor = {growth_factor}", xy=(0, last_entry['Confirmed cases']))

    ax.annotate(last_entry['Confirmed cases'], xy=(final_x, last_entry['Confirmed cases']))

    plt.show()

    

def concatenate(filter=None):

    confirmed_series = time_series(covid_19_confirmed, 'Confirmed cases', filter)

    deaths_series = time_series(covid_19_deaths, 'Deaths', filter)

    recovered_series = time_series(covid_19_recovered, 'Recovered cases', filter)



    series = [confirmed_series, deaths_series, recovered_series]

    return pd.concat(series, axis=1)



def show_data_for(country):

    if country == 'World':

        show_data(concatenate(), country)

    else:

        show_data(concatenate(lambda df: df['Country/Region'] == country), country)



show_data_for('World')

show_data_for('China')

show_data_for('US')

show_data_for('Italy')

show_data_for('Korea, South')