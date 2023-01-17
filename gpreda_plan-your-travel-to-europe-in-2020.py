import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt

import warnings

warnings.filterwarnings("ignore")

data_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data_df['Date'] = pd.to_datetime(data_df['ObservationDate'])

data_df['Country'] = data_df['Country/Region']



data_ct = data_df.sort_values(by = ['Country','Date'], ascending=False)

filtered_data_ct_last = data_ct.drop_duplicates(subset = ['Country'], keep='first')

data_ct_agg = data_ct.groupby(['Date', 'Country']).sum().reset_index()

data_ct_agg['Active'] = data_ct_agg['Confirmed'] - data_ct_agg['Deaths'] - data_ct_agg['Recovered']



def plot_time_variation_countries_group_of_features(df, countries, features,title):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(6,6,figsize=(20, 20))



    for country in countries:

        plt.subplot(6,6,i + 1)

        df_ = df[(df['Country']==country)] 

        df_['date'] = df_['Date'].apply(lambda x: x.timetuple().tm_yday)

        for feature in features:

            g = sns.lineplot(x="date", y=feature, data=df_,  label=feature)

        plt.title(f'{country}') 

        plt.xlabel('')

        plt.ylabel('')

        i = i + 1

    fig.suptitle(title, fontsize=18)

    plt.show()  

eu28plus_countries = ['Austria','Belgium','Italy','Latvia','Bulgaria','Lithuania','Croatia','Luxembourg','Cyprus',

                  'Malta','Czech Republic','Netherlands','Denmark','Poland','Estonia','Portugal','Finland','Romania',

                  'France','Slovakia','Germany','Slovenia','Greece','Spain','Hungary','Sweden','Ireland','Switzerland', 'Norway', 'Israel',

                     'Serbia', 'Montenegro', 'North Macedonia', 'UK', 'Russia', 'Iceland']

eu28plus_countries.sort()

features = ['Recovered', 'Active']

plot_time_variation_countries_group_of_features(data_ct_agg, eu28plus_countries, features, 'Recovered (cumulative) vs. Active - cases vs. day of year')