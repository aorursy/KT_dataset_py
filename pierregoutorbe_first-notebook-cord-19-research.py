# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import warnings 



from ipywidgets import interact

pd.options.display.max_colwidth = 200

warnings.filterwarnings('ignore')
# Load Data

coordinates = pd.read_csv("../input/latitude-and-longitude-for-every-country-and-state/world_country_and_usa_states_latitude_and_longitude_values.csv")

country_coordinates = coordinates[['country_code','latitude','longitude','country']]

state_coordinates = coordinates[['usa_state_code','usa_state_latitude','usa_state_longitude','usa_state']]

df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df['Country/Region'].replace(['Mainland China'], 'China',inplace=True)

df['Country/Region'].replace(['US'], 'United States',inplace=True)

df['Country'] = df['Country/Region']

df = df[df.ObservationDate==np.max(df.ObservationDate)]

todays_date = '3/15/2020'



# Mortality rate for every country in the dataset

df_deaths = pd.DataFrame(df.groupby('Country')['Deaths'].sum())

df_confirmed = pd.DataFrame(df.groupby('Country')['Confirmed'].sum())

df_confirmed['Deaths'] = df_deaths['Deaths']

df_global = df_confirmed

df_global['Mortality Rate'] = np.round((df_global.Deaths.values/df_global.Confirmed.values)*100,2)

df_global = df_global.reset_index()

df_global = df_global.merge(country_coordinates, left_on='Country', right_on='country')

df_global = df_global[['Country','Confirmed','Deaths','Mortality Rate','latitude','longitude','country_code']]

df_global.columns = ['Country','Confirmed','Deaths','Mortality Rate','Latitude','Longitude','Country_Code']

df_global.to_csv('/kaggle/working/global_covid19_mortality_rates.csv')



# Mortality rate for every state in the USA

df_usa = df[df['Country/Region']=='United States']

df_usa = df_usa[df_usa.ObservationDate==np.max(df_usa.ObservationDate)]

df_usa['State'] = df_usa['Province/State']

df_usa['Mortality Rate'] = np.round((df_usa.Deaths.values/df_usa.Confirmed.values)*100,2)

df_usa.sort_values('Mortality Rate', ascending= False).head(10)

df_usa = df_usa.merge(state_coordinates, left_on='State', right_on='usa_state')

df_usa['Latitude'] = df_usa['usa_state_latitude']

df_usa['Longitude'] = df_usa['usa_state_longitude']

df_usa = df_usa[['State','Confirmed','Deaths','Recovered','Mortality Rate','Latitude','Longitude','usa_state_code']]

df_usa.columns = ['State','Confirmed','Deaths','Recovered','Mortality Rate','Latitude','Longitude','USA_State_Code']

df_usa.to_csv('/kaggle/working/usa_covid19_mortality_rates.csv')
fig = px.choropleth(df_global, 

                    locations="Country", 

                    color="Confirmed", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,10000],

                    title='Global COVID-19 Infections as of '+todays_date)

fig.show()



fig = px.choropleth(df_global, 

                    locations="Country", 

                    color="Deaths", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,100],

                    title='Global COVID-19 Deaths as of '+todays_date)

fig.show()



fig = px.choropleth(df_global, 

                    locations="Country", 

                    color="Mortality Rate", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,10],

                    title='Global COVID-19 Mortality Rates as of '+todays_date)

fig.show()
fig = px.bar(df_global.sort_values('Confirmed',ascending=False)[0:10], 

             x="Country", 

             y="Confirmed",

             title='Global COVID-19 Infections as of '+todays_date)

fig.show()



fig = px.bar(df_global.sort_values('Deaths',ascending=False)[0:10], 

             x="Country", 

             y="Deaths",

             title='Global COVID-19 Deaths as of '+todays_date)

fig.show()



fig = px.bar(df_global.sort_values('Confirmed',ascending=False)[0:10], 

             x="Country", 

             y="Mortality Rate",

             title='Global COVID-19 Mortality Rates as of '+todays_date+' for Countries with Top 10 Most Confirmed')

fig.show()
##input_dir = PurePath('../input/CORD-19-research-challenge/2020-03-13')



sources = pd.read_csv('../input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')  

def doi_url(d): return d if d.startswith('doi.org') else f'doi.org/{d}'



sources.doi = sources.doi.fillna('').apply(doi_url)

SOURCES_COLS = ['title', 'abstract', 'doi', 'pubmed_id', 'publish_time', 'authors', 'journal', 'has_full_text']

sources[SOURCES_COLS]



def show_sources(ShowAllColumns=False):

    return sources if ShowAllColumns else sources[SOURCES_COLS]



interact(show_sources);
class ResearchPapers:

    

    def __init__(self, sources: pd.DataFrame):

        self.sources = sources

        

    def __getitem__(self, item):

        df = self.sources.iloc[item].to_frame().fillna('')

        df.columns = ['Value']

        return df

    

    def _repr_html_(self):

        return self.sources._repr_html_()



papers = ResearchPapers(sources)
papers[10000]
def count_ngrams(dataframe,column,begin_ngram,end_ngram):

    # adapted from https://stackoverflow.com/questions/36572221/how-to-find-ngram-frequency-of-a-column-in-a-pandas-dataframe

    word_vectorizer = CountVectorizer(ngram_range=(begin_ngram,end_ngram), analyzer='word')

    sparse_matrix = word_vectorizer.fit_transform(dataframe['title'].dropna())

    frequencies = sum(sparse_matrix).toarray()[0]

    most_common = pd.DataFrame(frequencies, 

                               index=word_vectorizer.get_feature_names(), 

                               columns=['frequency']).sort_values('frequency',ascending=False)

    most_common['ngram'] = most_common.index

    most_common.reset_index()

    return most_common



def word_cloud_function(df,column,number_of_words):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    word_string=str(popular_words_nonstop)

    wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=number_of_words,

                          width=1000,height=1000,

                         ).generate(word_string)

    plt.clf()

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()



def word_bar_graph_function(df,column,title,nvals):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    plt.barh(range(nvals), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:nvals])])

    plt.yticks([x + 0.5 for x in range(nvals)], reversed(popular_words_nonstop[0:nvals]))

    plt.title(title)

    plt.show()

    

    

three_gram = count_ngrams(sources,'title',3,3)

words_to_exclude = ["my","to","at","for","it","the","with","from","would","there","or","if","it","but","of","in","as","and",'NaN','dtype']
# show most frequent words in titles

plt.figure(figsize=(10,10))

word_bar_graph_function(sources,column='title', 

                        title='Most common words in the TITLES of the papers in the CORD-19 dataset',

                        nvals=40)
fig = px.bar(three_gram.sort_values('frequency',ascending=False)[0:10], 

             x="frequency", 

             y="ngram",

             title='Most Common 3-Words in Titles of Papers in CORD-19 Dataset',

             orientation='h')

fig.show()
plt.figure(figsize=(10,10))

word_cloud_function(sources,'title',5000)
# show most frequent words in titles

plt.figure(figsize=(10,10))

word_bar_graph_function(sources,column='abstract', 

                        title='Most common words in the Abstract of the papers in the CORD-19 dataset',

                        nvals=40)