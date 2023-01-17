from fuzzywuzzy import fuzz

from fuzzywuzzy import process



import re

import nltk

from nltk.util import ngrams

from nltk.collocations import BigramCollocationFinder

from nltk.metrics import BigramAssocMeasures

from nltk.stem.porter import *

from nltk.corpus import stopwords



from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets



import plotly.graph_objects as go

import plotly.express as px



from matplotlib import pyplot as plt

plt.style.use('ggplot')





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
accidents_df = pd.read_csv('/kaggle/input/airplane-crashes-since-1908/Airplane_Crashes_and_Fatalities_Since_1908.csv')

accidents_df['Date'] = pd.to_datetime(accidents_df['Date'])





accidents_df = accidents_df[~accidents_df.Type.isna()]

accidents_df['Type'] = accidents_df.Type.str.lower()

accidents_df = accidents_df[~accidents_df.Location.isna()]



accidents_df['Location'] =accidents_df.apply(lambda r: r['Location'].replace('AtlantiOcean', 'Atlantic Ocean'), axis=1)

accidents_df['Location'] =accidents_df.apply(lambda r: r['Location'].replace('PacifiOcean', 'Pacific Ocean'), axis=1)





aircraft_data = pd.read_csv('../input/aircraft-production-data/aircraft_data.csv', index_col='Unnamed: 0')

aircraft_data['aircraft'] = aircraft_data.aircraft.str.lower()
aircraft_names = aircraft_data.aircraft.to_list()
def get_similar_name(string, references=aircraft_names):

    max_ref = ''

    max_sim = 0

    

    for i, item in enumerate(references):

        sim = fuzz.ratio(item, string)

        if sim >= max_sim:

            max_ref = item

            max_sim = sim

    

    if max_sim < 50:

        return None, 0

    return max_ref, max_sim
get_similar_name('lockheed hercules')
#This is expensive (O(n^2))

accidents_df['MostSimilar'] = accidents_df.apply(lambda r: get_similar_name(r['Type'])[0], axis=1)
accidents_df.sort_values(by='Date').head(9)
accidents_df =  accidents_df[accidents_df['Date'] > '1950-01-01']

accidents_df = accidents_df[(~accidents_df.MostSimilar.isna()) & ~accidents_df.Operator.isna()].reset_index(drop=True)



accidents_df = accidents_df[~accidents_df.Operator.str.contains('Military')]
merged = pd.merge(accidents_df, aircraft_data, left_on='MostSimilar', right_on='aircraft', how='left')
accident_counts = merged.groupby('MostSimilar').agg(count=('MostSimilar', 'count')).sort_values(by='count', ascending=False).reset_index()

accident_counts.head(15)
accident_counts = merged.groupby('MostSimilar').agg(count=('MostSimilar', 'count'), built=('nbBuilt', 'mean')).sort_values(by='count', ascending=False).reset_index()

accident_counts['accidentRatio'] = accident_counts['count'] / accident_counts['built']



accident_counts = accident_counts[(accident_counts['built'] > accident_counts['count']) & (accident_counts['built'] > 50)]



accident_counts.sort_values(by='accidentRatio', ascending=False).head(30)
stemmer = PorterStemmer()

STOPWORDS = stopwords.words('english') + ['aircraft', 'plane']
def plot_common_words(dataframe):

    

    summaries = stemmer.stem(' '.join(dataframe.Summary.dropna().to_list())).replace('.','').replace(',','').split()

    summaries = [word for word in summaries if word not in STOPWORDS]

    

    word_fd = nltk.FreqDist(summaries)

    bigram_fd = nltk.FreqDist(nltk.bigrams(summaries))

    

    fig, ax = plt.subplots(1,2, figsize=(22,10))



    common = word_fd.most_common()[:20]

    ax[0].barh([i[0] for i in common], [i[1] for i in common])

    ax[0].invert_yaxis()



    common = bigram_fd.most_common()[:20]

    ax[1].barh([i[0][0] + ' ' + i[0][1] for i in common], [i[1] for i in common])

    ax[1].invert_yaxis()



    return fig
@interact(airplane_name=accident_counts.sort_values(by='accidentRatio', ascending=False).head(30).MostSimilar.to_list())

def plot_causes_airplane(airplane_name):

    fig = plot_common_words(merged[merged.aircraft == airplane_name])

    fig.suptitle(airplane_name)

    fig.show()   
# Or scatic version for kernel view-mode

top_names = accident_counts.sort_values(by='accidentRatio', ascending=False).head(30).MostSimilar.to_list()

for name in top_names[:3]:

    fig = plot_common_words(merged[merged.aircraft == name])

    fig.suptitle(name)

    fig.show()
# This is expensive (the requests to the service take time, around 1h) so I saved the result in a CSV to save time

if False:

    import geopy

    from time import sleep

    from tqdm import tqdm

    from geopy.geocoders import Nominatim



    geolocator = Nominatim(user_agent="aircraftaccidents")



    locations = []

    locnames = merged.Location.unique().tolist()



    with tqdm(total=len(locnames)) as pbar:

        for i, item in enumerate(locnames):

            loc = geolocator.geocode(item)

            if loc is not None:

                locations.append((loc.latitude, loc.longitude))

            else:

                locations.append((None, None))



            sleep(1)

            pbar.update(1)

    locs_df = pd.DataFrame(locations, columns=['lat', 'lon'])

    locs_df['name'] = locnames

    locs_df.to_csv('../input/locationcoordinates/locations.csv')

else:

    locs_df = pd.read_csv('../input/locationcoordinates/locations.csv')
merged = pd.merge(merged, locs_df, left_on='Location', right_on='name')
df = merged[merged.aircraft.isin(accident_counts.sort_values(by='accidentRatio', ascending=False).head(10).MostSimilar)]



fig = px.scatter_geo(df, 

                     lat="lat",

                     lon="lon",

                     color='aircraft',

                     text='Location'

                     )



fig.update_layout(

    title = 'Most dangerous aircraft crashes',

)



fig.show()