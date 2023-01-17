# imports

import os

import pandas as pd

import geopandas as gpd

from shapely import wkt

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.notebook import tqdm

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.probability import FreqDist

import re
root_dir = os.path.join('..', 'input', 'cdp-unlocking-climate-solutions')

main_paths = {'cities_disclosing': os.path.join(root_dir, 'Cities', 'Cities Disclosing'),

         'cities_questionnaires': os.path.join(root_dir, 'Cities', 'Cities Responses'),

         'corp_climate_change': os.path.join(root_dir, 'Corporations','Corporations Disclosing', 'Climate Change'),

         'corp_water_seq': os.path.join(root_dir, 'Corporations','Corporations Disclosing', 'Water Security'),

         'corp_quest_cc': os.path.join(root_dir, 'Corporations','Corporations Responses', 'Climate Change'),

         'corp_quest_cc': os.path.join(root_dir, 'Corporations','Corporations Responses', 'Water Security')

        }



dfs = {}

for k, v in main_paths.items():

    for f in os.listdir(v):

        df = pd.read_csv(os.path.join(v, f))

        dfs[k+'_'+f] = df
cities_dics = pd.concat(

    [dfs['cities_disclosing_2018_Cities_Disclosing_to_CDP.csv'],

    dfs['cities_disclosing_2019_Cities_Disclosing_to_CDP.csv'],

    dfs['cities_disclosing_2020_Cities_Disclosing_to_CDP.csv']])



cities_quest = pd.concat(

    [dfs['cities_questionnaires_2018_Full_Cities_Dataset.csv'],

    dfs['cities_questionnaires_2019_Full_Cities_Dataset.csv'],

    dfs['cities_questionnaires_2020_Full_Cities_Dataset.csv']])
cities_dics.head()
cities_dics.isna().sum()
cities_dics.describe()
pop = cities_dics[['Population', 'City', 'Country']].dropna().sort_values('Population', ascending=False)

top_10 = pd.DataFrame({'Population': pop.Population.values[0:10],

          'City': pop.City.values[0:10],

            'Country': pop.Country.values[0:10]})

bottom_10 = pd.DataFrame({'Population': pop.Population.values[-10:],

             'City': pop.City.values[-10:],

            'Country': pop.Country.values[-10:]})
plt.rcParams["figure.figsize"] = [11,5]

fig, ax = plt.subplots(1, 2)

ax[0].bar(x=top_10['City'], height=top_10['Population'], color='gold');

ax[1].bar(x=bottom_10['City'], height=bottom_10['Population'], color='aqua');

ax[0].tick_params(axis='x', rotation=55);

ax[1].tick_params(axis='x', rotation=55);

ax[0].set_title('The most populated cities');

ax[1].set_title('The least populated cities');
plt.rcParams["figure.figsize"] = [11,5]

fig, ax = plt.subplots(1, 2)

ax[0].bar(x=top_10['Country'], height=top_10['Population'], color='gold');

ax[1].bar(x=bottom_10['Country'], height=bottom_10['Population'], color='aqua');

ax[0].tick_params(axis='x', rotation=55);

ax[1].tick_params(axis='x', rotation=55);

ax[0].set_title('The most populated countries');

ax[1].set_title('The least populated countries');
del pop, top_10, bottom_10
cities_quest.head(3)
cities_quest.isna().sum()
print('We have {} % responses. Others are NaNs.'.format(round(

    100*cities_quest['Response Answer'].isna().sum()/len(cities_quest), 3)))
cities_quest = cities_quest[cities_quest['Response Answer'].notnull()]
cities_quest['Question Name'].value_counts()[:5]
cities_quest['Response Answer'].value_counts()[:10]
cities_quest['Response Answer'].value_counts()[-5:]
cities_quest[cities_quest['Response Answer']=='Question not applicable'].head(3)
def words_from_column(frame, column, sample=False):

    

    if sample:

        words = frame[column].sample(n=1000)

    else:

        words = frame[column]

        

    words = [re.sub(r'[^\w\s]', '', x).lower() for x in words]

    if len(words)==0:

        words = frame[column]

    words = tqdm([y for x in words for y in x.split(' ')])

    words = [x for x in words if x not in stopwords.words('english')]

    words = [x for x in words if x not in '`!@#$%^&*()_+=~".,?']

    

    return words
words = words_from_column(cities_quest, 'Question Name', sample=True)

print('Top 10 words in questions sample are:\n{}'.format(', '.join(words[:10])))
words = words_from_column(cities_quest[cities_quest['Row Name'].notnull()], 'Row Name', True)

print('Top 10 words in topics sample are:\n{}'.format(', '.join(words[:10])))
orgs = cities_quest[cities_quest['Response Answer'].notnull()].groupby('Organization')

top_responders = orgs.size().sort_values(ascending=False)[:10]

plt.bar(top_responders.index, top_responders.values, color='green');

plt.tick_params(axis='x', rotation=45);

plt.title('Top 10 responders by questions counts');
plt.rcParams["figure.figsize"] = [16,10]

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

ax = world.plot(color='white', edgecolor='black')

cities_dics['geometry'] = cities_dics['City Location'].dropna().reset_index(drop=True).apply(wkt.loads)

cities_geo = gpd.GeoDataFrame(cities_dics, geometry='geometry')

cities_geo.plot(color='green', ax=ax, markersize=5);
water_df = pd.concat(

    [dfs['corp_quest_cc_2018_Full_Water_Security_Dataset.csv'],

    dfs['corp_quest_cc_2019_Full_Water_Security_Dataset.csv'],

    dfs['corp_quest_cc_2020_Full_Water_Security_Dataset.csv']])

water_df.head(3)
water_df.shape
water_df.describe()
water_df.isna().sum()
print('Out of {} non-null values we have {} unique ones.'.format(

    water_df.comments.dropna().shape[0], water_df.comments.dropna().nunique()))
plt.rcParams["figure.figsize"] = [15, 7]

orgs = water_df[water_df['response_value'].notnull()].groupby('organization')

top_responders = orgs.size().sort_values(ascending=False)[:15]

plt.bar(top_responders.index, top_responders.values, color='#d52915');

plt.tick_params(axis='x', rotation=45);

plt.title('Top 10 responders by questions counts');