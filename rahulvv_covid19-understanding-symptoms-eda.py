# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the libraries



import matplotlib.pyplot as plt

%matplotlib inline



import folium

import seaborn as sns

sns.set(style='whitegrid')



import math



import nltk

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from wordcloud import WordCloud



import warnings

warnings.filterwarnings('ignore')



from itertools import cycle

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
line_list = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

line_list.head(3)
# checking the columns



line_list.columns
# many columns do not contain any info, and therefore deleted

# line_list['Unnamed: 23'].values

# array([nan, nan, nan, ..., nan, nan, nan])



# New dataframe contains 19 columns and 1085 data points



relevent_columns = line_list.columns[:19]

line_list = line_list[relevent_columns]



# Column 'Unnamed: 3' doesnt contain any relevent info and is deleted

line_list.drop(['Unnamed: 3'], axis=1, inplace=True)



# Checking the shape of dataframe

print(line_list.shape)
# Collecting all the datetime features



date_features = ['reporting date', 'symptom_onset', 'hosp_visit_date', 'exposure_start', 'exposure_end']



# Pasing all the date columns



for date_feature in date_features:

    line_list[date_feature] = pd.to_datetime(line_list[date_feature])
# Checking datatypes and shape



line_list.info()
# Checking for NaN values in Date features



for i, date_feature in enumerate(date_features, start=1):

    print('Statement {}: {} contains NaN values : {}'.format(i, date_feature, line_list[date_feature].isnull().any()))
# Making a seperate dataframe for all the date colums



date_df = line_list[date_features]

date_df
# extracting new information from the date columns and creating seperate dataframe for the new info derived



date_df['exposure_period'] = date_df.exposure_end - date_df.exposure_start

date_df['diagnose_period'] = date_df['reporting date'] - date_df.hosp_visit_date 

date_df['exp_to_symptm_period'] = date_df.symptom_onset - date_df.exposure_start

date_df['sympt_to_hosp_period'] = date_df.hosp_visit_date - date_df.symptom_onset



period_df = date_df[['exposure_period', 'diagnose_period', 'exp_to_symptm_period', 'sympt_to_hosp_period']]

period_df
# Understanding number of days elapsed per for different events



# Setting the titles for the graphs



titles = ['days exposed to virus', 'days taken to diagnose the disease', 'days elapsed from exposure to showing symptoms', 'days taken to visit the hospital after showing symptoms']



#creating subplots



fig, ax = plt.subplots(4, figsize=(10,20))

ax = ax.flatten()

fig.suptitle('Plotting number of days elapsed between different events', fontsize=15)

for i, period in enumerate(period_df.columns):

    days_elapsed = period_df[period].value_counts().rename_axis('days').reset_index(name='count')

    days_elapsed.sort_values('days', inplace=True)

    days_elapsed['days'] = days_elapsed['days'].astype('str')

    days_elapsed['days'] = days_elapsed['days'].str.split('00:00:00.', expand=True)

    sns.barplot(y='days', x='count', data=days_elapsed, ax=ax[i])

    ax[i].set_title(titles[i])

plt.tight_layout(pad=5)
open_line = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

open_line = open_line.iloc[:, :-12]

open_line.head(3)
# Check for missing information, datatypes and shape

open_line.info()
# running a loop to look at the data contained in each feature columns to analyze which one is relevent 



for col in open_line.columns:

    print(open_line[col].value_counts().head(4))

    print('')
# Our data frame reduces as shown below



open_line = open_line[['sex', 'city', 'province', 'country','latitude', 'longitude', 'geo_resolution',

                       'date_onset_symptoms', 'date_admission_hospital', 'date_confirmation', 'symptoms', 'lives_in_Wuhan', 

                       'travel_history_dates', 'travel_history_location']]

open_line.shape
# Removing the ambiguities in the 'sex' column



open_line.drop(open_line.loc[open_line.sex=='4000'].index, inplace=True)

open_line['sex'].replace('male', 'Male', inplace=True)

open_line['sex'].replace('female', 'Female', inplace=True)
# Plotting the distribution of gender



gender_df = open_line.sex.value_counts().rename_axis('gender').reset_index(name='count')

gender_df.head()

sns.barplot(y='gender', x='count', data=gender_df)

plt.title('Distribution of gender', fontsize=15)

plt.show()
# Separate dataframe for city latitude, longitude



location_df = open_line.groupby(['city'])['latitude', 'longitude'].mean().reset_index()

location_df = location_df.dropna()



# Geospatial tagging using folium



world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='cartodbpositron')



for lat, lon,city in zip(location_df['latitude'], location_df['longitude'], location_df['city']):

    folium.CircleMarker([lat, lon], radius=.5, color='red', fill_color='black',fill_opacity= 0.8).add_to(world_map)

world_map
# Seperate dataframe for symptoms



symptoms_df = open_line['symptoms']

symptoms_df.dropna(inplace=True)

symptoms_df = symptoms_df.reset_index()
# A function to extract symptoms



def find_symptoms(word):

    word_split = word.replace('()',',').split(',')

    word_split = [word.strip().rstrip(',') for word in word_split]

    key_symptoms.extend(word_split)
# creating a dataframe of major symptoms



key_symptoms = []

symptoms_df['symptoms'].dropna().apply(find_symptoms)

key_symptoms = pd.Series(key_symptoms)

key_symptoms = key_symptoms[key_symptoms!='']

major_symptoms = key_symptoms.value_counts()

major_symptoms[:10]
# A function to plot the major symptoms



def word_cloud(words):

    wordcloud = WordCloud(background_color='black', width = 1024, height=720).generate(words)

    plt.clf()

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis('off')

    plt.show()
# display using wordcloud



plt.figure(figsize=(15,9))

word_cloud(' '.join(major_symptoms.index.tolist()))
# to be continued

confirmed_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed_US.csv')

death_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths_US.csv')

confirmed_df.head()
death_df.head()
confirmed_df.columns
death_df.columns
confirmed_df.shape
confirmed_ts = confirmed_df[confirmed_df.columns[11:]].T

death_ts = death_df[death_df.columns[12:]].T
# Dataframe for the daily cumulative sum of confirmed cases

confirmed_ts['total_confirmed'] = confirmed_ts.sum(axis=1)

confirmed_ts = confirmed_ts['total_confirmed'].rename_axis('date').reset_index().set_index(['date'])



# Dataframe for the daily cumulative sum of death cases

death_ts['total_death'] = death_ts.sum(axis=1)

death_ts = death_ts['total_death'].rename_axis('date').reset_index().set_index(['date'])



# Concatenating both to get a time series dataframe

ts_df = pd.concat([confirmed_ts, death_ts], axis=1)
#Plotting the ocnfirmed and death cases 



fig, ax = plt.subplots(figsize=(18,6))

ax.plot(ts_df.total_confirmed, color=next(color_cycle), label='Total Confirmed', marker='o')

ax.plot(ts_df.total_death, color=next(color_cycle), label='Total deaths', marker='o')

ax.set_xticklabels(ts_df.index, rotation=90)

plt.legend(fancybox=True, ncol=3, fontsize=15, shadow=True)

plt.title('Confirmed cases in the United States')

plt.show()
# to be continued