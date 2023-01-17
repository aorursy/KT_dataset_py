# Data Processing Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns



# Module for parsing DateTime objects

from datetime import datetime



import locale # Used for formatting large number to have commas

locale.setlocale(locale.LC_ALL, '')



# Filepaths to datasets

maindata_filepath = "/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv"

linelist_filepath = "/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv"

openline_filepath = "/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv"

confirmed_filepath = "/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv"

recovered_filepath = "/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv"

deaths_filepath = "/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv"



# Read csv: covid_19_data.csv

covid = pd.read_csv(maindata_filepath,index_col=0)
print("Novel Corona Virus 2019 Dataset: Day level information on covid-19 affected cases")

print("Cumulative Data as of",covid['ObservationDate'].iloc[-1])



# Sub-dataframe: Contains latest data

filter_latest_data = covid['ObservationDate'] == covid['ObservationDate'].iloc[-1]

covid_latest = covid[filter_latest_data].copy()



print("Confirmed Cases:",int(covid_latest['Confirmed'].sum()))

print("Death Toll:", int(covid_latest['Deaths'].sum()), ", ", ((covid_latest['Deaths'].sum() / covid_latest['Confirmed'].sum())*100),"% of Confirmed Cases")

print("Recovered:", int(covid_latest['Recovered'].sum()), ", ", ((covid_latest['Recovered'].sum() / covid_latest['Confirmed'].sum())*100),"% of Confirmed Cases")

print("Source: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset")
# Top 10 Countries with most cases of COVID-19

latest_date = covid['ObservationDate'].iloc[-1]

covid_latest = covid[covid['ObservationDate'] == latest_date].groupby(['Country/Region']).sum().copy()

covid_latest = covid_latest.sort_values(by='Confirmed',ascending=False)

covid_latest[:10]
# Function for generating lineplot

def make_lineplot(country,x_dates=covid['ObservationDate'].unique()):

    '''

    Generates lineplot of cumulative counts of

    confirmed cases, deaths, and recoveries on COVID-19 patients

    country - str, Country/Region

    x_dates - list, array, or pandas series: Observation Dates

    '''



    # Filter data per country

    filter_country = covid['Country/Region'] == country

    covid_country = covid[filter_country].copy()



    # Dictionary of daily data

    confirmed_dict = dict()

    deaths_dict = dict()

    recovered_dict = dict()



    for date in x_dates:

        # filter dataframe by date

        filter_country_date = covid_country['ObservationDate'] == date

        covid_country_date = covid_country[filter_country_date].copy()

        short_date = date[:5]



        # Assign data to dictionary key: Confirmed

        confirmed_dict[short_date] = covid_country_date['Confirmed'].sum()



        # Assign data to dictionary key: Deaths

        deaths_dict[short_date] = covid_country_date['Deaths'].sum()



        # Assign data to dictionary key: Recovered

        recovered_dict[short_date] = covid_country_date['Recovered'].sum()



    # Initialize the matplotlib figure

    f, ax = plt.subplots(figsize=(25,5))



    # Plot the confirmed cases

    sns.set_color_codes('pastel')

    sns.lineplot(x=list(confirmed_dict.keys()),

                y=list(confirmed_dict.values()),

                label='Confirmed',

                color='green'

                )



    # Plot the deaths

    sns.set_color_codes('pastel')

    sns.lineplot(x=list(deaths_dict.keys()),

                y=list(deaths_dict.values()),

                label='Deaths',

                color='red'

                )



    # Plot the recoveries

    sns.set_color_codes('pastel')

    sns.lineplot(x=list(recovered_dict.keys()),

                y=list(recovered_dict.values()),

                label='Recovered',

                color='blue'

                )



    # Add a legend and informative axis label

    ax.legend(ncol=3, loc='lower right', frameon=True)

    ax.set(xlabel='Observation Dates',

           ylabel='Cumulative Counts',

           title=f"COVID-19 Data ({country})"

          )
# Line plot for Mainland China

country = 'Mainland China'

make_lineplot(country)



# Latest Statistics

# Filter data per country

filter_country = covid['Country/Region'] == country

covid_country = covid[filter_country].copy()



print("Novel Corona Virus 2019 Dataset: Day level information on covid-19 affected cases")

print("Cumulative Data as of", covid_country['ObservationDate'].iloc[-1],"from",covid_country['Country/Region'].unique()[0])



filter_country_latest = covid_country['ObservationDate'] == covid_country['ObservationDate'].iloc[-1]

covid_country_latest = covid_country[filter_country_latest].copy()



print("Confirmed Cases:",int(covid_country_latest['Confirmed'].sum()))

print("Death Toll:", int(covid_country_latest['Deaths'].sum()), ", ", ((covid_country_latest['Deaths'].sum() / covid_country_latest['Confirmed'].sum())*100),"% of Confirmed Cases")

print("Recovered:", int(covid_country_latest['Recovered'].sum()), ", ", ((covid_country_latest['Recovered'].sum() / covid_country_latest['Confirmed'].sum())*100),"% of Confirmed Cases")

print("Source: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset")
# Line plot for USA

country = 'US'

make_lineplot(country)



# Latest Statistics

# Filter data per country

filter_country = covid['Country/Region'] == country

covid_country = covid[filter_country].copy()



print("Novel Corona Virus 2019 Dataset: Day level information on covid-19 affected cases")

print("Cumulative Data as of", covid_country['ObservationDate'].iloc[-1],"from",covid_country['Country/Region'].unique()[0])



filter_country_latest = covid_country['ObservationDate'] == covid_country['ObservationDate'].iloc[-1]

covid_country_latest = covid_country[filter_country_latest].copy()



print("Confirmed Cases:",int(covid_country_latest['Confirmed'].sum()))

print("Death Toll:", int(covid_country_latest['Deaths'].sum()), ", ", ((covid_country_latest['Deaths'].sum() / covid_country_latest['Confirmed'].sum())*100),"% of Confirmed Cases")

print("Recovered:", int(covid_country_latest['Recovered'].sum()), ", ", ((covid_country_latest['Recovered'].sum() / covid_country_latest['Confirmed'].sum())*100),"% of Confirmed Cases")

print("Source: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset")
# Line plot for South

country = 'South Korea'

make_lineplot(country)



# Latest Statistics

# Filter data per country

filter_country = covid['Country/Region'] == country

covid_country = covid[filter_country].copy()



print("Novel Corona Virus 2019 Dataset: Day level information on covid-19 affected cases")

print("Cumulative Data as of", covid_country['ObservationDate'].iloc[-1],"from",covid_country['Country/Region'].unique()[0])



filter_country_latest = covid_country['ObservationDate'] == covid_country['ObservationDate'].iloc[-1]

covid_country_latest = covid_country[filter_country_latest].copy()



print("Confirmed Cases:",int(covid_country_latest['Confirmed'].sum()))

print("Death Toll:", int(covid_country_latest['Deaths'].sum()), ", ", ((covid_country_latest['Deaths'].sum() / covid_country_latest['Confirmed'].sum())*100),"% of Confirmed Cases")

print("Recovered:", int(covid_country_latest['Recovered'].sum()), ", ", ((covid_country_latest['Recovered'].sum() / covid_country_latest['Confirmed'].sum())*100),"% of Confirmed Cases")

print("Source: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset")
# Line plot for South

country = 'Malaysia'

make_lineplot(country)



# Latest Statistics

# Filter data per country

filter_country = covid['Country/Region'] == country

covid_country = covid[filter_country].copy()



print("Novel Corona Virus 2019 Dataset: Day level information on covid-19 affected cases")

print("Cumulative Data as of", covid_country['ObservationDate'].iloc[-1],"from",covid_country['Country/Region'].unique()[0])



filter_country_latest = covid_country['ObservationDate'] == covid_country['ObservationDate'].iloc[-1]

covid_country_latest = covid_country[filter_country_latest].copy()



print("Confirmed Cases:",int(covid_country_latest['Confirmed'].sum()))

print("Death Toll:", int(covid_country_latest['Deaths'].sum()), ", ", ((covid_country_latest['Deaths'].sum() / covid_country_latest['Confirmed'].sum())*100),"% of Confirmed Cases")

print("Recovered:", int(covid_country_latest['Recovered'].sum()), ", ", ((covid_country_latest['Recovered'].sum() / covid_country_latest['Confirmed'].sum())*100),"% of Confirmed Cases")

print("Source: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset")
# Line plot for Philippines

country = 'Philippines'

make_lineplot(country)



# Latest Statistics

# Filter data per country

filter_country = covid['Country/Region'] == country

covid_country = covid[filter_country].copy()



print("Novel Corona Virus 2019 Dataset: Day level information on covid-19 affected cases")

print("Cumulative Data as of", covid_country['ObservationDate'].iloc[-1],"from",covid_country['Country/Region'].unique()[0])



filter_country_latest = covid_country['ObservationDate'] == covid_country['ObservationDate'].iloc[-1]

covid_country_latest = covid_country[filter_country_latest].copy()



print("Confirmed Cases:",int(covid_country_latest['Confirmed'].sum()))

print("Death Toll:", int(covid_country_latest['Deaths'].sum()), ", ", ((covid_country_latest['Deaths'].sum() / covid_country_latest['Confirmed'].sum())*100),"% of Confirmed Cases")

print("Recovered:", int(covid_country_latest['Recovered'].sum()), ", ", ((covid_country_latest['Recovered'].sum() / covid_country_latest['Confirmed'].sum())*100),"% of Confirmed Cases")

print("Source: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset")