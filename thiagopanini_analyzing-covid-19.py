# Libs to be used on this project

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns

%matplotlib inline

from datetime import datetime

from warnings import filterwarnings

filterwarnings('ignore')
# Formatting matplotlib figures

def format_spines(ax, right_border=True):

    """

    This function sets up borders from an axis and personalize colors

    

    Input:

        Axis and a flag for deciding or not to plot the right border

    Returns:

        Plot configuration

    """    

    # Setting up colors

    ax.spines['bottom'].set_color('#CCCCCC')

    ax.spines['left'].set_color('#CCCCCC')

    ax.spines['top'].set_visible(False)

    if right_border:

        ax.spines['right'].set_color('#CCCCCC')

    else:

        ax.spines['right'].set_color('#FFFFFF')

    ax.patch.set_facecolor('#FFFFFF')
# Reading the updated corona virus dataset

df_corona = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df_corona.columns = [c.lower().replace(' ', '_').replace('/', '_') for c in df_corona.columns]

df_corona.head()
# Transforming date columns

df_corona['last_update_cleaned'] = pd.to_datetime(df_corona['last_update']).dt.date

df_corona['obs_date_cleaned'] = pd.to_datetime(df_corona['observationdate']).dt.date

df_corona.drop(['last_update', 'observationdate'], axis=1, inplace=True)

df_corona.columns = ['sno', 'province_state', 'country_region', 'confirmed', 

                     'deaths', 'recovered', 'observation_date', 'last_update']

df_corona.head()
# Observation and update dates

print(f'Range of observation date: from {df_corona["observation_date"].min()} to {df_corona["observation_date"].max()}\n')

print(f'Range of update date: from {df_corona["last_update"].min()} to {df_corona["last_update"].max()}')
# Virus evolution on Mainland China

main_china = df_corona.query('country_region == "Mainland China"')



# Grouping data

cols_group = ['last_update', 'confirmed', 'deaths', 'recovered']

corona_sum = df_corona.groupby(by=['last_update'], as_index=False).sum().loc[:, cols_group]

china_sum = main_china.groupby(by=['last_update', 'country_region'], 

                               as_index=False).sum().loc[:, cols_group + ['country_region']]



# Showing confirmed cases of corona virus on Mainland China

fig, ax = plt.subplots(figsize=(15, 6))

sns.lineplot(x='last_update', y='confirmed', data=corona_sum, ax=ax, color='darkslateblue', label='World', marker='o')

sns.lineplot(x='last_update', y='confirmed', data=china_sum, ax=ax, color='crimson', label='China', marker='X')



# Making some annotations

x_highlight = corona_sum['last_update'][22]

y_highlight = corona_sum['confirmed'][22]

highlight_perc_increase = 100 * (corona_sum['confirmed'][21] / corona_sum['confirmed'][22])

ax.annotate(f'Huge increase\nof {highlight_perc_increase:.2f}% from \nthe day before', 

            (mdates.date2num(x_highlight), y_highlight), 

            xytext=(25, -25),textcoords='offset points', arrowprops=dict(arrowstyle='-|>', fc='w'), color='dimgrey')

plt.vlines(x_highlight, 0, y_highlight, linestyle="dashed", color='silver')

    

# Making another annotations

ax.fill_between(corona_sum['last_update'], corona_sum['confirmed'], china_sum['confirmed'], color='silver', alpha=.5)

ax.annotate(f'Virus spreading\naround the World ', 

            (mdates.date2num(corona_sum['last_update'][-1:]), corona_sum['confirmed'][-1:]), 

            xytext=(-70, -75), textcoords='offset points', 

            bbox=dict(boxstyle="round4", fc="w"), color='dimgrey',

            arrowprops=dict(arrowstyle='-|>', connectionstyle="arc3, rad=1.7", fc="w"))





# Finishing plot

ax.set_title('Evolution of Corona Virus Confirmed Cases', size=14, color='dimgrey', pad=20)

ax.set_ylim(ymin=0)

format_spines(ax, right_border=False)

plt.xticks(rotation=90)

plt.xticks(corona_sum['last_update'])

plt.show()
# How we are doing against the Virus? Is there any evolution of recovering cases?

fig, ax = plt.subplots(figsize=(15, 6))

colors = ['darkslateblue', 'crimson', 'lightseagreen']

corona_sum.drop('last_update', axis=1).div(corona_sum.sum(1).astype(float), 

                                           axis=0).plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.7)

format_spines(ax, right_border=False)

plt.xticks(np.arange(0, len(corona_sum)), corona_sum['last_update'])

ax.set_title('Percentage of Confirmed, Deaths and Recovered Cases', size=14, color='dimgrey')

plt.show()