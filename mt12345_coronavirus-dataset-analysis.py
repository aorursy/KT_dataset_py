"""

Kaggle project: Analysis of COVID-19 CSV data

module and data imports

"""

import datetime

import os



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



pd.plotting.register_matplotlib_converters()



%matplotlib inline

sns.set()

plt.rcParams['figure.figsize'] = (10,6)



data_dir = '/kaggle/input/novel-corona-virus-2019-dataset'



# relate the list indices (from the search above), to sensible hash table values

data_files = {

    'confirmed': 'time_series_covid_19_confirmed.csv',

    'recovered': 'time_series_covid_19_recovered.csv',

    'complete': 'covid_19_data.csv',

    'deaths': 'time_series_covid_19_deaths.csv',

}



DF = { key: pd.read_csv(os.path.join(data_dir, data_files[key])) for key in data_files}

DF['complete'] = DF['complete'].astype({'ObservationDate': 'Datetime64', 'Last Update': 'Datetime64'}).set_index('SNo') # cast

complete = DF['complete'] # shortcut (main dataset)

# some provinces are empty, others are filled with country/region -> ensure consistency

complete['Province/State'].fillna(complete['Country/Region'], inplace=True)

complete.set_index(['Country/Region', 'Province/State'], inplace=True)
def plot_country(country_name: str, log_scale: bool = False) -> list:

    """

    Plot a line chart of cases in a give country

    """

    country_cases = RAW_CASE_DATA['complete'].loc[country_name]

    # any regions with null values -> fill with the country name

    country_cases.index = country_cases.index.fillna(country_name)

    ax=list()

    ax.append(sns.lineplot(x='ObservationDate', y='Confirmed', data=country_cases))

    ax.append(sns.lineplot(x='ObservationDate', y='Deaths', data=country_cases))

    ax.append(sns.lineplot(x='ObservationDate', y='Recovered', data=country_cases))

    ax[0].axes.set_xlim(right=datetime.date.today())

    plt.suptitle('{} COVID-19 Cases'.format(country_name.title()))

    plt.xlabel('Date')

    plt.xticks(rotation=45)

    plt.ylabel('Cases')

    return ax





def country_summary(country_name: str):

    """

    Return key statistics for a specified countries

    """

    subframe = RAW_CASE_DATA['complete'].loc[country_name]

    return {'cases': subframe['Confirmed'].max(),

    'deaths': subframe['Deaths'].max(),

    'recoveries': subframe['Recovered'].max(),

    'prevalence': subframe['Confirmed'].max() - subframe['Recovered'].max() - subframe['Deaths'].max()

    }
plot_country('Australia')

country_summary('Australia')