!pip install datefinder

!pip install geotext

!pip install langdetect
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datefinder

import pycountry

from geotext import GeoText

import covid19_tools as cv19

import re

from collections import Counter

import geopandas as gpd

import json

import matplotlib.pyplot as plt

from langdetect import detect



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
print('Loading full text')

full_text_repr = cv19.load_full_text(meta_df,

                                     '../input/CORD-19-research-challenge')
def get_body_text(full_text_repr):

    body_text = []

    for article in full_text_repr:

        text = [body_text['text'] for body_text in article['body_text']]

        body_text.append(''.join(text))

    return body_text



body_text_repr = get_body_text(full_text_repr)



full_text_ids = [article['paper_id'] for article in full_text_repr]

meta_df['full_text'] = None

meta_df['full_text'] = meta_df['sha'].apply(lambda x: full_text_ids.index(x) if x in full_text_ids else -1)

meta_df['full_text'] = meta_df['full_text'].apply(lambda x: body_text_repr[x] if x != -1 else None)
print(meta_df.shape)

meta_df.head()
def find_country(text):

    '''

    Extracts countries using pycountry.

    '''

    entities = []

    

    for country in pycountry.countries:

        if country.name in text and country not in entities:

            entities.append(country.name)

    return entities



#Adds a new column in the df containing the countries found in the titles.

meta_df["country_title"] = np.nan

meta_df['country_title'] = meta_df[meta_df['title'].notna()]['title'].apply(lambda x: find_country(x))



#Adds a new column in the df containing the countries found in the abstracts.

meta_df["country_abstract"] = np.nan

meta_df['country_abstract'] = meta_df[meta_df['abstract'].notna()]['abstract'].apply(lambda x: find_country(x))



#Adds a new column in the df containing the countries found in the full text.

meta_df["country_text"] = np.nan

meta_df['country_text'] = meta_df[meta_df['full_text'].notna()]['full_text'].apply(lambda x: find_country(x))

meta_df.head()
def list_entities_no_duplicates(df, column):

    '''

    Extracts a list of unique entities (i.e. countries or cities).

    '''

    entities_list = df[df[column].notna()][column].values.tolist()

    entities_list_flat = [item for sublist in entities_list for item in sublist]

    entities_list_flat_no_dupli = list(set(entities_list_flat))

    return entities_list_flat_no_dupli



#Convert title countries into a list to validate them

countries_title_list = list_entities_no_duplicates(meta_df, 'country_title')

print(countries_title_list)



#Convert abstract countries into a list to validate them

countries_abstract_list = list_entities_no_duplicates(meta_df, 'country_abstract')

print(countries_abstract_list)



#Convert text countries into a list to validate them

countries_text_list = list_entities_no_duplicates(meta_df, 'country_text')

print(countries_text_list)
def find_city(text):

    '''

    Extracts cities using geotext.

    '''

    places = GeoText(text)

    

    return places.cities



#Adds a new column in the df containing the country found in the titles.

meta_df["city_title"] = np.nan

meta_df['city_title'] = meta_df[meta_df['title'].notna()]['title'].apply(lambda x: find_city(x))



#Adds a new column in the df containing the country found in the titles.

meta_df["city_abstract"] = np.nan

meta_df['city_abstract'] = meta_df[meta_df['abstract'].notna()]['abstract'].apply(lambda x: find_city(x))



#Adds a new column in the df containing the country found in the titles.

meta_df["city_text"] = np.nan

meta_df['city_text'] = meta_df[meta_df['full_text'].notna()]['full_text'].apply(lambda x: find_city(x))

meta_df.head()
#Convert title cities into a list to validate them

cities_title_list = list_entities_no_duplicates(meta_df, 'city_title')

print(cities_title_list)



#Convert title locations into a list to validate them

cities_abstract_list = list_entities_no_duplicates(meta_df, 'city_abstract')

print(cities_abstract_list)



#Convert text locations into a list to validate them

cities_abstract_list = list_entities_no_duplicates(meta_df, 'city_text')

print(cities_abstract_list)
def find_location(df, id_to_search):

    '''

    Given a df and an id it returns the countries and cities found in the title and abstract of the article the id belongs to.

    '''

    country_title = df[df['cord_uid'] == id_to_search]['country_title']

    country_abstract = df[df['cord_uid'] == id_to_search]['country_abstract']

    city_title = df[df['cord_uid'] == id_to_search]['city_title']

    city_abstract = df[df['cord_uid'] == id_to_search]['city_abstract']

        

    return country_title, country_abstract, city_title, city_abstract



# example

xqhn0vbp = find_location(meta_df, 'xqhn0vbp')

print(xqhn0vbp)
meta_df.to_csv('/kaggle/working/df_locations.csv', sep = ';', index = False)
meta_df.columns
meta_df.shape
# Load the countries shapefile

shapefile = '/kaggle/input/worldmapshapes/ne_110m_admin_0_countries.shp'



#Read shapefile using Geopandas

gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]



#Rename columns

gdf.columns = ['country', 'country_code', 'geometry']

gdf["country"] = gdf["country"].replace('United States of America', 'United States') 

gdf.head()
print(gdf[gdf['country'] == 'Antarctica'])

#Drop row corresponding to 'Antarctica'

gdf = gdf.drop(gdf.index[159])
def count_entities(df, column):

    '''

    Counts entities (i.e., countries or cities) and returns a dataframe with entities and counts.

    '''

    entities_list = df[df[column].notna()][column].values.tolist()

    entities_list_flat = [item for sublist in entities_list for item in sublist]

    count = Counter(entities_list_flat)

    

    df_counts = pd.DataFrame(count.items(), columns=[column, 'count'])

    

    return df_counts



count_countries_title = count_entities(meta_df, 'country_title')

print(count_countries_title)



count_countries_abstract = count_entities(meta_df, 'country_abstract')

print(count_countries_abstract)



count_countries_text = count_entities(meta_df, 'country_text')

print(count_countries_text)
#Perform left merge to preserve every row in gdf with countries extracted from titles

merged_country_title = gdf.merge(count_countries_title, left_on = 'country', right_on = 'country_title', how = 'left')

merged_country_title.to_csv('/kaggle/working/merged_country_title.csv', sep = ';', index = False)

print(merged_country_title.head())



#Perform left merge to preserve every row in gdf with countries extracted from abstracts

merged_country_abstract = gdf.merge(count_countries_abstract, left_on = 'country', right_on = 'country_abstract', how = 'left')

merged_country_abstract.to_csv('/kaggle/working/merged_country_abstract.csv', sep = ';', index = False)

print(merged_country_abstract.head())



#Perform left merge to preserve every row in gdf with countries extracted from full text

merged_country_text = gdf.merge(count_countries_text, left_on = 'country', right_on = 'country_text', how = 'left')

merged_country_text.to_csv('/kaggle/working/merged_country_text.csv', sep = ';', index = False)

print(merged_country_text.head())
# set a variable that will call whatever column we want to visualise on the map

variable = 'count'



# set the range for the choropleth

vmin, vmax = 0, 1400



# create figure and axes for Matplotlib

fig, ax = plt.subplots(1, figsize=(100, 60))



# create map

merged_country_title.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')



# remove the axis

ax.axis('off')



# add a title

ax.set_title('Choropleth map of articles published by country, countries extracted from titles', fontdict={'fontsize': '100', 'fontweight' : '3'})



# Create colorbar as a legend



norm = plt.Normalize(vmin=vmin, vmax=vmax)

sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)



# empty array for the data range

sm._A = []



# add the colorbar to the figure

cbar = fig.colorbar(sm, orientation="horizontal", shrink=0.3);

cbar.ax.tick_params(labelsize=50)
# set a variable that will call whatever column we want to visualise on the map

variable = 'count'



# set the range for the choropleth

vmin, vmax = 0, 3000



# create figure and axes for Matplotlib

fig, ax = plt.subplots(1, figsize=(100, 60))



# create map

merged_country_abstract.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')



# remove the axis

ax.axis('off')



# add a title

ax.set_title('Choropleth map of articles published by country, countries extracted from abstracts', fontdict={'fontsize': '100', 'fontweight' : '3'})



# Create colorbar as a legend



norm = plt.Normalize(vmin=vmin, vmax=vmax)

sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)



# empty array for the data range

sm._A = []



# add the colorbar to the figure

cbar = fig.colorbar(sm, orientation="horizontal", shrink=0.3);

cbar.ax.tick_params(labelsize=50)
# set a variable that will call whatever column we want to visualise on the map

variable = 'count'



# set the range for the choropleth

vmin, vmax = 0, 10000



# create figure and axes for Matplotlib

fig, ax = plt.subplots(1, figsize=(100, 60))



# create map

merged_country_text.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')



# remove the axis

ax.axis('off')



# add a title

ax.set_title('Choropleth map of articles published by country, countries extracted from full text', fontdict={'fontsize': '100', 'fontweight' : '3'})



# Create colorbar as a legend



norm = plt.Normalize(vmin=vmin, vmax=vmax)

sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)



# empty array for the data range

sm._A = []



# add the colorbar to the figure

cbar = fig.colorbar(sm, orientation="horizontal", shrink=0.3);

cbar.ax.tick_params(labelsize=50)
def extract_date(text):

    new_text = re.sub(r'\(.*?\)', "", text) # Exclude text in parenthesis

    search = re.findall(r'\D(\d{4})\D', new_text)

  

    years = [search for year in search if year <= '2020']

    

    return years



#Adds a new column in the df containing the years found in the titles.

meta_df["year_title"] = np.nan

meta_df['year_title'] = meta_df[meta_df['title'].notna()]['title'].apply(lambda x: extract_date(x))

meta_df.head()
#Adds a new column in the df containing the language the title is written

meta_df["lang"] = np.nan

meta_df['lang'] = meta_df[meta_df['title'].notna()]['title'].apply(lambda x: detect(x))

meta_df.head()
meta_df.to_csv('/kaggle/working/df_updated.csv', sep = ';', index = False)