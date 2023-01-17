!pip install transformers

!pip install torch

!pip install pyshorteners

!pip install xlrd

!pip install wikipedia
%matplotlib inline

import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

import json

from pandas.io.json import json_normalize

from wordcloud import WordCloud, STOPWORDS







import warnings

warnings.filterwarnings("ignore")









import pandas as pd

import numpy as np

import nltk

import re

import datetime



#UrlLib for http handlings

from bs4 import BeautifulSoup

import urllib.request

import re



import urllib.request  

import bs4 as BeautifulSoup

import nltk



#from string import punctuation

#from nltk.corpus import stopwords

#from nltk.tokenize import word_tokenize

#from nltk.tokenize import sent_tokenize



from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import urllib.request  

import bs4 as BeautifulSoup

import nltk

from string import punctuation

import wikipedia



#For the next version 

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



import torch

import json 

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


df               = pd.read_csv('../input/ted-talks/ted_main.csv')

transcript_df    = pd.read_csv('../input/ted-talks/transcripts.csv')

ultimate_TED     = pd.read_csv('../input/ted-ultimate-dataset/2020-05-01/ted_talks_it.csv')

updated_TED      = pd.read_csv('../input/ted-talks-dataset/ted_main.csv')

TED_Talks_AI     = pd.read_excel('../input/ted-talks-20062020/Artificial Intelligence_TED_Talks.xlsx', sheet_name='Sheet1') 
TED_Talks_AI
import datetime

df['film_date'] = df['film_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))

df['published_date'] = df['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
df.tail(1)
updated_TED.tail(1)
df = df[['name', 'title', 'description', 'main_speaker', 'speaker_occupation', 'num_speaker', 'duration', 'event', 'film_date', 'published_date', 'comments', 'tags', 'languages', 'ratings', 'related_talks', 'url', 'views']]
df.head(1)
updated_TED.head(1)
updated_TED.rename(columns = {'speaker_name':'main_speaker', 'title':'title', 'posted_date':'published_date','duration':'duration', 'Link':'url', 'about_speaker':'speaker_occupation','about_talk':'description', 'views':'views', 'tags':'tags'}, inplace = True) 
updated_TED.head(1)
df.shape
updated_TED.shape
#df_row_reindex = pd.concat([df, updated_TED], ignore_index=True)



#df_row_reindex.tail()
#values = {'num_speaker': 1, 'event': " ", 'film_date': 1140825600, 'published_date': 1140825600, 'comments': 0, 'tags': "TED", 'languages': 0, 'ratings': 0, 'related_talks': " ", 'views': 0}

#df_row_reindex.fillna(value=values)
df.tail()
len(df)
df.dtypes
pop_talks = df[['title', 'main_speaker', 'views', 'film_date']].sort_values('views', ascending=False)[:15]

pop_talks
pop_talks['abbr'] = pop_talks['main_speaker'].apply(lambda x: x[:3])

sns.set_style("whitegrid")

plt.figure(figsize=(10,6))

sns.barplot(x='abbr', y='views', data=pop_talks)
sns.distplot(df['views'])
sns.distplot(df[df['views'] < 0.4e7]['views'])
df['views'].describe()
df['comments'].describe()
sns.distplot(df['comments'])
sns.distplot(df[df['comments'] < 500]['comments'])
sns.jointplot(x='views', y='comments', data=df)
df[['views', 'comments']].corr()
df[['title', 'main_speaker','views', 'comments']].sort_values('comments', ascending=False).head(10)
df['dis_quo'] = df['comments']/df['views']
df[['title', 'main_speaker','views', 'comments', 'dis_quo', 'film_date']].sort_values('dis_quo', ascending=False).head(10)
df['month'] = df['film_date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1])



month_df = pd.DataFrame(df['month'].value_counts()).reset_index()

month_df.columns = ['month', 'talks']
sns.barplot(x='month', y='talks', data=month_df, order=month_order)
df_x = df[df['event'].str.contains('TEDx')]

x_month_df = pd.DataFrame(df_x['month'].value_counts().reset_index())

x_month_df.columns = ['month', 'talks']
sns.barplot(x='month', y='talks', data=x_month_df, order=month_order)
def getday(x):

    day, month, year = (int(i) for i in x.split('-'))    

    answer = datetime.date(year, month, day).weekday()

    return day_order[answer]
df['day'] = df['film_date'].apply(getday)
day_df = pd.DataFrame(df['day'].value_counts()).reset_index()

day_df.columns = ['day', 'talks']
sns.barplot(x='day', y='talks', data=day_df, order=day_order)
df['year'] = df['film_date'].apply(lambda x: x.split('-')[2])

year_df = pd.DataFrame(df['year'].value_counts().reset_index())

year_df.columns = ['year', 'talks']
plt.figure(figsize=(18,5))

sns.pointplot(x='year', y='talks', data=year_df)
months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
hmap_df = df.copy()

hmap_df['film_date'] = hmap_df['film_date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1] + " " + str(x.split('-')[2]))

hmap_df = pd.pivot_table(hmap_df[['film_date', 'title']], index='film_date', aggfunc='count').reset_index()

hmap_df['month_num'] = hmap_df['film_date'].apply(lambda x: months[x.split()[0]])

hmap_df['year'] = hmap_df['film_date'].apply(lambda x: x.split()[1])

hmap_df = hmap_df.sort_values(['year', 'month_num'])

hmap_df = hmap_df[['month_num', 'year', 'title']]

hmap_df = hmap_df.pivot('month_num', 'year', 'title')

hmap_df = hmap_df.fillna(0)
f, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(hmap_df, annot=True, linewidths=.5, ax=ax, fmt='n', yticklabels=month_order)

speaker_df = df.groupby('main_speaker').count().reset_index()[['main_speaker', 'comments']]

speaker_df.columns = ['main_speaker', 'appearances']

speaker_df = speaker_df.sort_values('appearances', ascending=False)

speaker_df.head(10)
occupation_df = df.groupby('speaker_occupation').count().reset_index()[['speaker_occupation', 'comments']]

occupation_df.columns = ['occupation', 'appearances']

occupation_df = occupation_df.sort_values('appearances', ascending=False)
plt.figure(figsize=(15,5))

sns.barplot(x='occupation', y='appearances', data=occupation_df.head(10))

plt.show()
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))

sns.boxplot(x='speaker_occupation', y='views', data=df[df['speaker_occupation'].isin(occupation_df.head(10)['occupation'])], palette="muted", ax =ax)

ax.set_ylim([0, 0.4e7])

plt.show()
df['num_speaker'].value_counts()
df[df['num_speaker'] == 5][['title', 'description', 'main_speaker', 'event']]
events_df = df[['title', 'event']].groupby('event').count().reset_index()

events_df.columns = ['event', 'talks']

events_df = events_df.sort_values('talks', ascending=False)

events_df.head(10)
df['languages'].describe()
df[df['languages'] == 72]
sns.jointplot(x='languages', y='views', data=df)

plt.show()
import ast

df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x))
s = df.apply(lambda x: pd.Series(x['tags']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'theme'
theme_df = df.drop('tags', axis=1).join(s)

theme_df.head()
len(theme_df['theme'].value_counts())
pop_themes = pd.DataFrame(theme_df['theme'].value_counts()).reset_index()

pop_themes.columns = ['theme', 'talks']

pop_themes.head(10)
plt.figure(figsize=(15,5))

sns.barplot(x='theme', y='talks', data=pop_themes.head(10))

plt.show()
pop_theme_talks = theme_df[(theme_df['theme'].isin(pop_themes.head(8)['theme'])) & (theme_df['theme'] != 'TEDx')]

pop_theme_talks['year'] = pop_theme_talks['year'].astype('int')

pop_theme_talks = pop_theme_talks[pop_theme_talks['year'] > 2008]
themes = list(pop_themes.head(8)['theme'])

themes.remove('TEDx')

ctab = pd.crosstab([pop_theme_talks['year']], pop_theme_talks['theme']).apply(lambda x: x/x.sum(), axis=1)

ctab[themes].plot(kind='bar', stacked=True, colormap='rainbow', figsize=(12,8)).legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
ctab[themes].plot(kind='line', stacked=False, colormap='rainbow', figsize=(12,8)).legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
pop_theme_talks = theme_df[theme_df['theme'].isin(pop_themes.head(10)['theme'])]

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))

sns.boxplot(x='theme', y='views', data=pop_theme_talks, palette="muted", ax =ax)

ax.set_ylim([0, 0.4e7])
#Convert to minutes

df['duration'] = df['duration']/60

df['duration'].describe()
df[df['duration'] == 2.25]
df[df['duration'] == 87.6]
sns.jointplot(x='duration', y='views', data=df[df['duration'] < 25])

plt.xlabel('Duration')

plt.ylabel('Views')

plt.show()
df2 = pd.read_csv('../input/ted-talks/transcripts.csv')

df2.head()
len(df2)
df3 = pd.merge(left=df,right=df2, how='left', left_on='url', right_on='url')

df3.head()
df3['transcript'] = df3['transcript'].fillna('')

df3['wc'] = df3['transcript'].apply(lambda x: len(x.split()))
df3['wc'].describe()
df3['wpm'] = df3['wc']/df3['duration']

df3['wpm'].describe()
df3[df3['wpm'] > 245]
sns.jointplot(x='wpm', y='views', data=df3[df3['duration'] < 25])

plt.show()
df.iloc[1]['ratings']
df['ratings'] = df['ratings'].apply(lambda x: ast.literal_eval(x))
df['funny'] = df['ratings'].apply(lambda x: x[0]['count'])

df['jawdrop'] = df['ratings'].apply(lambda x: x[-3]['count'])

df['beautiful'] = df['ratings'].apply(lambda x: x[3]['count'])

df['confusing'] = df['ratings'].apply(lambda x: x[2]['count'])

df.head()
df[['title', 'main_speaker', 'views', 'published_date', 'funny']].sort_values('funny', ascending=False)[:10]
df[['title', 'main_speaker', 'views', 'published_date', 'beautiful']].sort_values('beautiful', ascending=False)[:10]
df[['title', 'main_speaker', 'views', 'published_date', 'jawdrop']].sort_values('jawdrop', ascending=False)[:10]
df[['title', 'main_speaker', 'views', 'published_date', 'confusing']].sort_values('confusing', ascending=False)[:10]
df['related_talks'] = df['related_talks'].apply(lambda x: ast.literal_eval(x))
s = df.apply(lambda x: pd.Series(x['related_talks']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'related'
related_df = df.drop('related_talks', axis=1).join(s)

related_df['related'] = related_df['related'].apply(lambda x: x['title'])
d = dict(related_df['title'].drop_duplicates())

d = {v: k for k, v in d.items()}
related_df['title'] = related_df['title'].apply(lambda x: d[x])

related_df['related'] = related_df['related'].apply(lambda x: d[x])
related_df = related_df[['title', 'related']]

related_df.head()
edges = list(zip(related_df['title'], related_df['related']))
import networkx as nx

G = nx.Graph()

G.add_edges_from(edges)
plt.figure(figsize=(25, 25))

nx.draw(G, with_labels=False)
corpus = ' '.join(df2['transcript'])

corpus = corpus.replace('.', '. ')
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpus)

plt.figure(figsize=(12,15))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
device = torch.device('cpu')

model = T5ForConditionalGeneration.from_pretrained('t5-base')

tokenizer = T5Tokenizer.from_pretrained('t5-base')
i=0



n = int(transcript_df['url'].count())

for i in range(n):



    transcript_df['url'][i] = transcript_df['url'][i].replace("\n", "/transcript")

    print(transcript_df['url'][i])

    

    i=i+1
transcript_df
def summarization_links_t5(d,i):

    import urllib.request

    import re

    

    # Data collection from the links using web scraping(using Urllib library)

    links_url = d['url'].tolist()

    links_url = links_url[i]

    links_url = links_url

    #links_url = "hhttps://www.ted.com/talks/sir_ken_robinson_do_schools_kill_creativity/"





    text = urllib.request.urlopen(links_url)

    

    global summary

    summary = ''

    link_summary = text.read()

    link_summary

    

    # Parsing the URL content 

    link_parsed = BeautifulSoup.BeautifulSoup(link_summary,'html')

    

    # Returning <p> tags

    paragraphs = link_parsed.find_all('p')

    

    # To get the content within all poaragrphs loop through it

    link_content = ''

    for p in paragraphs:  

        link_content += p.text

    

    # Removing Square Brackets and Extra Spaces

    link_content = re.sub(r'\[[0-9]*\]', ' ', link_content)

    link_content = re.sub(r'\s+', ' ', link_content)

    

    # Removing special characters and digits

    formatted_article_text = re.sub('[^a-zA-Z]', ' ', link_content )

    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    

    text = formatted_article_text

    #text_temp = text.read()



    preprocess_text = text.strip().replace("\n","")

    t5_prepared_Text = "summarize: "+preprocess_text

    #print ("original text preprocessed: \n", preprocess_text)



    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)





    # summmarize 

    summary_ids = model.generate(tokenized_text,

                                    num_beams=4,

                                    no_repeat_ngram_size=2,

                                    min_length=100,

                                    max_length=600,

                                    early_stopping=True)



    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)



    #print ("\n\nSummarized text: \n",output)

    return output
summarization_links_t5(transcript_df,0)
print()

i=0

d =transcript_df

n = int(d["url"].count()) - 2465

print("Here you can find the summary of "+ str(n) + " TED Talks. It would take hours to run everything and I'm coding at a very late hour.")

for i in range(n):

    print("----------------------------------------------------------------------------------------------------------------------")

    print(str(i)+") "+ summarization_links_t5(d,i) +".")

    print("Talk: "+ d["url"][i])

    print(" ")

    

    i=i+1
TED_Talks_AI.head()
transcript_df.head()
TED_Talks_AI.rename({'Title_URL': 'url'}, axis=1, inplace=True)
TED_Talks_AI['url'][1]
TED_Talks_AI['transcript'] = ""



TED_Talks_AI['transcript'][1] = TED_Talks_AI['url'][1] + "/transcript/"
TED_Talks_AI['transcript'][1]
i=0



n = int(TED_Talks_AI['url'].count())

for i in range(n):



    TED_Talks_AI['url'][i] = TED_Talks_AI['url'][i] + "/transcript/"

    print(TED_Talks_AI['url'][i])

    

    i=i+1
summarization_links_t5(TED_Talks_AI,0)