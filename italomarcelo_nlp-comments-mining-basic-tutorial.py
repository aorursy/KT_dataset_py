import pandas as pd

import numpy as np

import requests

!pip install feedparser

import feedparser

from bs4 import BeautifulSoup

from textblob import TextBlob

import re, string

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer 

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

# update textblob

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

nltk.download('stopwords')

nltk.download('wordnet')
rssurl = 'https://forums.macrumors.com/threads/kuo-apple-to-accelerate-adoption-of-mini-led-displays-in-ipad-and-mac-notebook-lineups.2255816/'
article = requests.get(rssurl)

articles = BeautifulSoup(article.content, 'html.parser')

print('Subject: ', articles.title.string)

print('Url: ', rssurl)
rows = []

for pc in articles.find_all("div", {"class": "bbWrapper"})[1:]:

  row=''

  for txt in pc.strings:

    row += txt

  rows.append(row)
# setup cores

colors = ['PowderBlue', 'Tomato', 'MidnightBlue', 'Goldenrod', 'MediumOrchid',

          'Salmon', 'Lime', 'PapayaWhip', 'DeepSkyBlue', 'LightPink']



# setup estrutura de graficos

# pie

explode = (0.01, 0.01)

labels = ['Positive', 'Negative']



pos = 0

neg = 0

for r in rows:

  txt = TextBlob(r)

  polarity = txt.sentiment.polarity

  if polarity != 0:

    if polarity > 0:

        pos += 1

    else:

        neg += 1
plt.figure(figsize=(15,8))

plt.pie([pos, neg], labels=labels, colors=colors, startangle=90, explode = explode, autopct = '%1.2f%%')

plt.axis('equal') 

plt.title(articles.title.string)

plt.show()


df = pd.DataFrame()

df['comments'] = rows

df.head()
swords = stopwords.words('english')

def cleaningText(txt):

  txt = txt.lower() # lowercase

  txt = re.sub('@','',txt) # remove @ 

  txt = re.sub('\[.*\]','',txt) # remove contents between brackets

  txt = re.sub('<.*?>+','',txt) # remove contents between less and more signs

  txt = re.sub('https?://\S+|www\.\S+', '', txt) # remove URLs

  txt = re.sub(re.escape(string.punctuation), '', txt) # remove punctuation

  txt = re.sub(r'[^a-zA-Z ]+', '', txt) # remove numbers

  txt = re.sub('\n', '', txt) # remove line break

  txt = nltk.word_tokenize(txt) # creating a word's list

  txt = [word for word in txt if word not in swords] 



  return txt
test = 'Today our Economy Professionals talking about "Brazilian Economy Estrategy". See the video here https://www.1.com'

test = cleaningText(test)

test
df.comments = df.comments.apply(lambda c: cleaningText(c))

df.head()
def lemmatization(txt):

    txt = [WordNetLemmatizer().lemmatize(i) for i in txt]

    txt = [WordNetLemmatizer().lemmatize(i, 'v') for i in txt]

    return txt

  

test2 = lemmatization(test)

test2
df.comments = df.comments.apply(lambda l: lemmatization(l))

df.head()
# creating list with all words frm dataframe

wordsDF = []

for listdf in df.comments.to_list():

  for w in listdf:

    wordsDF.append(w)
fig, ax1 = plt.subplots(sharey=True, figsize=(15,9))

sns.barplot(x=pd.Series(wordsDF).value_counts()[:20].index, 

            y=pd.Series(wordsDF).value_counts()[:20].values,

            ax=ax1).set_title('Top 20')

plt.xlabel('word')

plt.ylabel('count')

plt.xticks(rotation=80)
bwords = []

for i in nltk.bigrams(wordsDF):

    bwords.append(i)
bigrams = []



for b in bwords:

  bigrams.append(b)

print('Mean Bigrams: ', pd.Series(bigrams).value_counts().mean())

fig, ax1 = plt.subplots(sharey=True, figsize=(15,9))

sns.barplot(x=pd.Series(bwords).value_counts()[:20].index, 

            y=pd.Series(bwords).value_counts()[:20].values,

            ax=ax1).set_title('Top 20 bigrams')

plt.xlabel('brigram')

plt.ylabel('count')

plt.xticks(rotation=80)
pos = 0

neg = 0

for r in range(0, len(df)):

  txt = TextBlob(' '.join(df.comments[r]))

  polarity = txt.sentiment.polarity

  if polarity != 0:

    if polarity > 0:

        pos += 1

    else:

        neg += 1
fig, ax1 = plt.subplots(sharey=True, figsize=(10,5))

sns.barplot(x=labels, palette=colors,

            y=[pos, neg],

            ax=ax1).set_title('Polarity '+articles.title.string)

plt.xlabel('Status')

plt.ylabel('Qty')

plt.xticks(rotation=80)
wordcloud = WordCloud(width = 1400, height = 800, 

                background_color ='white', 

                min_font_size = 10).generate(' '.join(wordsDF))



# plot the WordCloud image                       

plt.figure(figsize = (15, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 