import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib as mpl

from matplotlib import pyplot as plt

import seaborn as sns

import glob

from collections import Counter

import matplotlib.pyplot as plot

import seaborn as sb

from matplotlib import cm

import json

import glob

import pandas as panda

import matplotlib.pyplot as plot

import re

import nltk

from nltk.sentiment import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk import sent_tokenize, word_tokenize

from wordcloud import WordCloud

import wordcloud

import datetime
PLOT_COLORS = ["#268bd2", "#0052CC", "#FF5722", "#b58900", "#003f5c"]

pd.options.display.float_format = '{:.2f}'.format

sns.set(style="ticks")

plt.rc('figure', figsize=(8, 5), dpi=100)

plt.rc('axes', labelpad=20, facecolor="#ffffff", linewidth=0.4, grid=True, labelsize=14)

plt.rc('patch', linewidth=0)

plt.rc('xtick.major', width=0.2)

plt.rc('ytick.major', width=0.2)

plt.rc('grid', color='#9E9E9E', linewidth=0.4)

plt.rc('font', family='Arial', weight='400', size=10)

plt.rc('text', color='#282828')

plt.rc('savefig', pad_inches=0.3, dpi=300)
path = r'../input/youtube-new'

all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:

    df = pd.read_csv(filename, index_col='video_id', encoding='latin-1')

    df['country'] = filename[21:23] #Ajout du pays

    li.append(df)        

my_data = pd.concat(li)
# Supression des valeurs null

my_data["description"] = my_data["description"].fillna(value="")



# Changement des types

## Dates

my_data['tresnding_date']=pd.to_datetime(my_data['trending_date'],format='%y.%d.%m')

my_data['publish_time']=pd.to_datetime(my_data['publish_time'],format='%Y-%m-%dT%H:%M:%S.%fZ')

my_data['publish_time'].dt.date

my_data.insert(4,'publish_date',my_data['publish_time'].dt.date)

my_data['publish_time']=my_data['publish_time'].dt.time

my_data[['publish_time','publish_date','tresnding_date']].head()

##String et Integer

type_int=['views','likes','dislikes','comment_count']

for i in type_int:

    my_data[i]=my_data[i].astype(int)

my_data['category_id']=my_data['category_id'].astype(str)
category_name_id = {}

with open('../input/youtube-new/US_category_id.json','r') as f:

    data=json.load(f)

    for category in data['items']:

        category_name_id[category['id']]=category['snippet']['title']

#category_name_id



my_data.insert(4,'category',my_data['category_id'].map(category_name_id))

my_data[['category_id','category']].head()
# Description des données numeriques

my_data.describe()
# Correlation des données du dataset

h_labels = [x.replace('_', ' ').title() for x in 

            list(my_data.select_dtypes(include=['number', 'bool']).columns.values)]



fig, ax = plt.subplots(figsize=(10,6))

_ = sns.heatmap(df.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)



#corr_matrix=my_data[['views', 'likes', 'dislikes', 'comment_count']].corr()

#corr_matrix
# Likes et Vues

sns.scatterplot(x='likes',y='views',data=my_data)
# Likes et nomb re de commentaires

sns.scatterplot(x='comment_count',y='views',data=my_data)
# Nombre de vues par catégorie

category_count=my_data['category'].value_counts()

category_count
# Les tags les plus utilisés 

tag=[]

for i in my_data['tags']:

    for j in i.split('|'):

        tag.append(j.replace('"'," ").strip().replace('"',""))

TAGS=pd.DataFrame(tag,columns=['TAGS'])        

TAGS=TAGS['TAGS'].value_counts().iloc[:6]

TAGS=TAGS.drop('[none]')

TAGS
# Taille des titres de videos

my_data["title_length"] = my_data["title"].apply(lambda x: len(x))



fig, ax = plt.subplots()

_ = sns.distplot(my_data["title_length"], kde=False, rug=False, 

                 color=PLOT_COLORS[3], hist_kws={'alpha': 1}, ax=ax)

_ = ax.set(xlabel="Title Length", ylabel="No. of videos", xticks=range(0, 110, 10))
# Catégories les plus populaires

def best_cat(data,title):

    plot.figure(figsize=(15,10))

    sb.set_style("whitegrid")

    ax = sb.barplot(y=data['index'],x=data['category'], data=data,orient='h')

    plot.xlabel("Nombre de videos")

    plot.ylabel("Categories")

    plot.title(title)



data = my_data[my_data['country']!='']['category'].value_counts().reset_index()

title = "\n Categories les plus populaires \n"

best_cat(data, title)
# Catègories par pays 

def category_per_country(list_categories_per_country, title):

    plot.yticks(rotation=30, fontsize=20)

    plot.xticks(rotation=30, fontsize=15)

    plot.title(title, fontsize=20)

    plot.legend(handlelength=5, fontsize=15, loc='center left', bbox_to_anchor=(1, 0.5))

    plot.show()



list_categories_per_country = my_data.reset_index().groupby(["country", "category"]).count()['video_id'].unstack().plot.barh(figsize=(12, 10), stacked=True)

title = "\n Catégories populaire par pays \n"

category_per_country(list_categories_per_country, title)
def polar_draw(sentiment, title):

    plot.figure(figsize=(16,10))

    sb.set(style="white",context="talk")

    ax = sb.barplot(x=sentiment['polarity'],y=sentiment['category'], data=sentiment,orient='h',palette="RdBu")

    plot.xlabel("polarity")

    plot.ylabel("categories")

    plot.title(title)
MAX_N = 1000

polarities = list()

en_stopwords = list(stopwords.words('english'))

de_stopwords = list(stopwords.words('german'))   

fr_stopwords = list(stopwords.words('french'))   

fr_stopwords.extend(de_stopwords)

fr_stopwords.extend(en_stopwords)



category_list = my_data['category'].unique()

for cat in category_list:

    print(cat)

    tags_word = my_data[my_data['category'] ==cat]['tags'].str.lower().str.cat(sep=' ')    

    tags_word = re.sub('[^A-Za-z]+', ' ', tags_word)

    word_tokens = word_tokenize(tags_word)

    filtered_sentence = [w for w in word_tokens if not w in fr_stopwords]

    without_single_chr = [word for word in filtered_sentence if len(word) > 2]

    cleaned_data_title = [word for word in without_single_chr if not word.isdigit()] 

    word_dist = nltk.FreqDist(cleaned_data_title)

    hnhk = pd.DataFrame(word_dist.most_common(MAX_N),

                    columns=['Word', 'Frequency'])



    compound = .0

    for word in hnhk['Word'].head(MAX_N):

        compound += SentimentIntensityAnalyzer().polarity_scores(word)['compound']



    polarities.append(compound)
category_list = pd.DataFrame(my_data['category'].unique())

polarities = pd.DataFrame(polarities)

sentiment = pd.concat([category_list,polarities],axis=1)

sentiment.columns = ['category','polarity']

sentiment=sentiment.sort_values('polarity').reset_index()

title = "Polarite des categories dans les videos Youtube"

polar_draw(sentiment, title)
# Combien de titres vidéo tendance contiennent un mot en majuscule?

def contains_capitalized_word(s):

    for w in s.split():

        if w.isupper():

            return True

    return False

my_data["contains_capitalized"] = my_data["title"].apply(contains_capitalized_word)

value_counts = my_data["contains_capitalized"].value_counts().to_dict()

fig, ax = plt.subplots()

_ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'], 

           colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'}, startangle=45)

_ = ax.axis('equal')

_ = ax.set_title('Vidéos contient un mot en majuscule?')
# Quel sont mes tags les plus populaires ?

def wcloud(data,bgcolor):

    plot.figure(figsize = (100,60))

    cloud = WordCloud(background_color = bgcolor, max_words = 60, 

                     width=1200, height=500,colormap="tab20b")

    cloud.generate(' '.join(data))

    plot.imshow(cloud)

    plot.axis('off')
fr_stopwords = list(stopwords.words('french'))   

tags_word = my_data['tags'].str.lower().str.cat(sep=' ')

tags_word = re.sub('[^A-Za-z]+', ' ', tags_word)

word_tokens = word_tokenize(tags_word)

filtered_sentence = [w for w in word_tokens if not w in fr_stopwords]

without_single_chr = [word for word in filtered_sentence if len(word) > 2 ]

cleaned_data_title = [word for word in without_single_chr if not word.isdigit()]



wcloud(cleaned_data_title,'white')