!pip install --upgrade -v pandas==1.0.1 --quiet
import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go

from fbprophet import Prophet

import pycountry

import plotly.express as px



from google.cloud import bigquery



print('Pandas version: {}'.format(pd.__version__))
def load_weather_data():

    %%time

    bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos.noaa_gsod")



    client = bigquery.Client()

    dataset_ref = client.dataset("noaa_gsod", project="bigquery-public-data")

    dataset = client.get_dataset(dataset_ref)



    tables = list(client.list_tables(dataset))



    table_ref = dataset_ref.table("stations")

    table = client.get_table(table_ref)

    stations_df = client.list_rows(table).to_dataframe()



    table_ref = dataset_ref.table("gsod2020")

    table = client.get_table(table_ref)

    twenty_twenty_df = client.list_rows(table).to_dataframe()



    stations_df['STN'] = stations_df['usaf'] + '-' + stations_df['wban']

    twenty_twenty_df['STN'] = twenty_twenty_df['stn'] + '-' + twenty_twenty_df['wban']



    cols_1 = ['STN', 'mo', 'da', 'temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']

    cols_2 = ['STN', 'country', 'state', 'call', 'lat', 'lon', 'elev']

    weather_df = twenty_twenty_df[cols_1].join(stations_df[cols_2].set_index('STN'), on='STN')



    weather_df.tail(10)
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['ObservationDate','Last Update']).sort_values(by='ObservationDate').fillna('')

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country', 'Province/State': 'Region'}, inplace=True)

df.drop(columns=['SNo', 'Deaths', 'Recovered'], inplace=True)

display(df.info())

display(df.sample(5))



print("External Data")

print(f"Earliest Entry: {df['Date'].min()}")

print(f"Last Entry:     {df['Date'].max()}")

print(f"Total Days:     {df['Date'].max() - df['Date'].min()}")
def plot_rate_of_change(df, country=None, region=None, grouping=['Date']):

    

    if country is None:

        selected_df = df

    elif region is None:

        selected_df = df.query('Country == "' + country + '"')

    else:

        selected_df = df.query('Country == "' + country + '" and Region == "' + region + '"' )

        

    country = 'Global' if country is None else country

    region = 'ALL' if region is None else region

    print('Country: {}'.format(country))

    print('Regions: {}'.format(region))



    grouped = selected_df.groupby(grouping).sum().reset_index().set_index('Date')

    grouped['Confirmed_ROC'] = grouped[['Confirmed']].pct_change().fillna(0) * 100.0

    grouped['SMA_3'] = grouped[['Confirmed_ROC']].rolling('3d').mean() 

    grouped['SMA_7'] = grouped[['Confirmed_ROC']].rolling('7d').mean()

    grouped['SMA_14'] = grouped[['Confirmed_ROC']].rolling('14d').mean()

    



    grouped[['Confirmed_ROC', 'SMA_3', 'SMA_7', 'SMA_14']].plot(title='Daily Rate of Change for: ' + country + ' - ' + region,

                                                       figsize=(20,8))

    plt.ylabel('pct%')   

    plt.show()

    display(pd.concat([grouped.head(1), grouped.tail(1)]))



plot_rate_of_change(df, 'US')

plot_rate_of_change(df, 'US', 'North Carolina')
plot_rate_of_change(df)
plot_rate_of_change(df, 'US')
plot_rate_of_change(df, 'Italy')
plot_rate_of_change(df, 'Mainland China')
plot_rate_of_change(df, 'Hong Kong')
plot_rate_of_change(df, 'South Korea')
plot_rate_of_change(df, 'US', 'California')

plot_rate_of_change(df, 'US', 'New York')

plot_rate_of_change(df, 'US', 'North Carolina')

plot_rate_of_change(df, 'US', 'Ohio')

plot_rate_of_change(df, 'US', 'Florida')