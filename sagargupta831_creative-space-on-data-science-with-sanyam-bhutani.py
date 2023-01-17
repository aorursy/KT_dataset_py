from IPython.display import Image

Image("../input/ctds-poster/ctds_1.png")
import plotly

import plotly.express as pv

import numpy as np

import pandas as pd

import seaborn as sns

sns.set(style ="whitegrid")

import matplotlib.pyplot as plt

from matplotlib import rcParams

import plotly.graph_objects as go

figg = go.Figure()

import re

import string

import nltk

from nltk.corpus import stopwords

from collections import Counter

from wordcloud import WordCloud

from collections import Counter

import warnings

warnings.filterwarnings('ignore')

import operator







CTDS_episodes = pd.read_csv('../input/chai-time-data-science/Episodes.csv',parse_dates=['recording_date','release_date'])



YouTube_thumbnails = pd.read_csv('../input/chai-time-data-science/YouTube Thumbnail Types.csv')



Anchor_thumbnails = pd.read_csv('../input/chai-time-data-science/Anchor Thumbnail Types.csv')



Description_details = pd.read_csv('../input/chai-time-data-science/Description.csv')
length = len(CTDS_episodes["episode_id"])

print(f'Total number of episodes Till Now : {length}','\n')

length1 = len(set(CTDS_episodes['heroes_location']))

print(f'Total number of countries from where  Guest belongs : {length1}','\n')

country = list(set(CTDS_episodes['heroes_location']))

print(f'list of countries are given here below :\n \n{ country }','\n')
CTDS_episodes.head()
# check for missing values in 'heroes_gender' column

gender = CTDS_episodes["heroes_gender"].isnull().sum()

print(f'Missing values in "heroes_gender" column is : {gender}')
CTDS_episodes['heroes_gender'].value_counts()
#plotting

rcParams["figure.figsize"] = 10,8

sns.countplot(x = CTDS_episodes["heroes_gender"],hue=CTDS_episodes["heroes_gender"])
#check for missing values in category column



category = CTDS_episodes["category"].isnull().sum()

print(f'Missing values in "category" column is : {category}')
CTDS_episodes['category'].value_counts()
#plotting



rcParams["figure.figsize"] = 10,8

sns.countplot(x = CTDS_episodes["category"],hue = CTDS_episodes["heroes_gender"]) 
fig = pv.bar(data_frame=CTDS_episodes,x ="flavour_of_tea",color="flavour_of_tea",title = "Bar_Graph : ' Distribution Of Flavour of Tea'")

fig.show()
rcParams["figure.figsize"] = 20,10

sns.countplot(hue = CTDS_episodes["recording_time"],x = CTDS_episodes["flavour_of_tea"])

fig = pv.bar(CTDS_episodes,x = "heroes_location", title="Bar_Graph : ML heroes location info check")

fig.show()
#counting ml heroes and identifying how many are from USA.



location_data = Counter(CTDS_episodes["heroes_location"])

sorted_location_data = dict(sorted(location_data.items(), key=operator.itemgetter(1),reverse=True))

fig = pv.funnel_area(names = list(sorted_location_data.keys()),values = list(sorted_location_data.values()),title="Heroes Location in Percentage") 

fig.show()
fig = pv.bar(CTDS_episodes,x = "heroes_location",color = 'recording_time', title="Bar_Graph : Recording Time Analysis")

fig.show()
#checking for missing subscribers in dataset. 

ys = CTDS_episodes["youtube_subscribers"].isnull().sum()

print(f'Missing values in "youtube_subscribers" column is : {ys}')
#sorting youtube subscribers in descending order.

highest_subs = CTDS_episodes.sort_values(by="youtube_subscribers",ascending=False)

rcParams["figure.figsize"] = 20,10

fig = pv.bar(x="heroes",y="youtube_subscribers",data_frame=highest_subs[:10],color = 'youtube_views',title="Number of Subscribers of CTDS show on Youtube")

fig.show()
#sorting youtube impression views in descending order.

highest_views = CTDS_episodes.sort_values(by="youtube_impression_views",ascending=False)

rcParams["figure.figsize"] = 20,10

fig = pv.bar(x="heroes",y="youtube_impression_views",color = 'release_date',data_frame=highest_views[:10],title="Number of impression views of ML heroes")

fig.show()
#sorting youtube likes in descending order.

highest_likes = CTDS_episodes.sort_values(by="youtube_likes",ascending=False)

rcParams["figure.figsize"] = 20,10

fig = pv.bar(x="episode_id",y="youtube_likes",color= 'heroes',data_frame=highest_views[:10],title="Number of YouTube Likes of CTDS.show on YouTube (Episodes comparison) ")

fig.show()
rcParams["figure.figsize"] = 20,10

fig = pv.line(CTDS_episodes,y= "youtube_dislikes" ,x = 'episode_id', title="YouTube Episode's dislike graph ")

fig.show()

fig = pv.line(data_frame=CTDS_episodes,x = "release_date",y="youtube_subscribers",title="Growth Rate of Subscribers w.r.t Release date on youtube channel")

fig.show()
# total subscribers on youtube channel

tot_subs = CTDS_episodes['youtube_subscribers'].sum()

print(f'Total subcscribers on youtube channel is : {tot_subs}' )
fig = pv.bar(CTDS_episodes,y = "episode_duration",x = "episode_id",color = 'category', title="Episode duration distribution w.r.t category and episode ID")

fig.show()
#sorting and analysing through visualising the graph

highest_ctr = CTDS_episodes.sort_values(by = 'youtube_ctr',ascending = False)



pv.bar(data_frame=highest_ctr,x="episode_id",y="youtube_ctr",color = 'category',title="bar_graph : 'Episodes with CTR'")
aw_duration = CTDS_episodes.sort_values(by = 'youtube_avg_watch_duration',ascending = False)

pv.bar(data_frame= aw_duration,x="episode_id",y="youtube_avg_watch_duration",color = 'category',title="bar_graph : 'Average Watch Duration")
fig = pv.area(data_frame=CTDS_episodes,x="release_date",y="anchor_plays",title =' Anchor plays according to release date')

fig.show()
fig = pv.bar(data_frame=CTDS_episodes.sort_values(by = 'anchor_plays',ascending = False)[:25],x="heroes",y="anchor_plays",title = 'Anchor plays according to Heroes name',color = 'anchor_plays')

fig.show()
fig = pv.bar(data_frame=CTDS_episodes.sort_values(by = 'spotify_listeners',ascending = False)[:25],x="heroes",y="spotify_listeners",title = 'Analysing Spotify listeners w.r.t heroes name',color = 'spotify_listeners')

fig.show()
fig = pv.bar(data_frame=CTDS_episodes.sort_values(by = 'apple_listeners',ascending = False)[:25],x="heroes",y="apple_listeners",title = 'Analysing Apple listeners w.r.t heroes name',color = 'apple_listeners')

fig.show()
fig = pv.bar(data_frame=CTDS_episodes.sort_values(by = ['apple_avg_listen_duration'],ascending = False),x="episode_id",y="apple_avg_listen_duration",title = 'Analysing Apple Average Listen duration w.r.t Episodes',color = 'category')

fig.show()
Description_details.head(10)
def cleaner(text):

    text = text.lower()

    text = re.sub('\w*\d\w*', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    return text

Description_details["description"] = Description_details["description"].apply(cleaner)
def text_preprocessing(text):

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    nopunc = cleaner(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    complete_txt = ' '.join(remove_stopwords)

    return complete_txt
Description_details["description"] = Description_details["description"].apply(text_preprocessing)

Description_details.head(10)
Description_details['new_list'] = Description_details['description'].apply(lambda x:str(x).split())

top_words = Counter([item for subset in Description_details['new_list'] for item in subset])

temp_data = pd.DataFrame(top_words.most_common(20))

temp_data.columns = ['Common_words','count']

temp_data.style.background_gradient(cmap='YlOrBr')

fig = pv.bar(temp_data, x="count", y="Common_words", title='Most Commmon Words in Description Data',color='Common_words')

fig.show()
def generate_word_cloud(text):

    wordcloud = WordCloud(width = 2000,height = 1000,background_color = 'lightyellow').generate(str(text))

    fig = plt.figure(figsize = (30, 20),facecolor = 'k',edgecolor = 'k')

    plt.imshow(wordcloud, interpolation = 'bilinear')

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.show()

description_text = Description_details.description[:100].values

generate_word_cloud(description_text)
