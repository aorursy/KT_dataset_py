import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import wordcloud



import matplotlib.pyplot as plt

import plotly.offline as py

from plotly.offline import init_notebook_mode

import plotly.graph_objs as go

import plotly.figure_factory as ff

init_notebook_mode(connected=True)

df = pd.read_csv('../input/Jerus20k.csv', encoding='ISO-8859-1')

df.head(5)
print('Number of rows: {}'.format(df.shape[0]))

print('Number of columns: {}'.format(df.shape[1]))

print('Data description')

print(df.describe())
df = df.sort_values('retweetCount', ascending=False)

text = df.iloc[0]['text']



print('Most retweeted tweet with {} retweets:' .format(max(df['retweetCount'])))

print(text)
df = df.sort_values('favoriteCount', ascending=False)

text = df.iloc[0]['text']

print('Most favorited tweet with {} favorites:' .format(max(df['favoriteCount'])))

print(text)
values = df['screenName'].value_counts()

values.head(5)
data = [go.Histogram(x=df['created'])]

py.iplot(data)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()



df['polarities'] = df['text'].apply(sid.polarity_scores)

df[['compound', 'neg', 'neu', 'pos']] = df['polarities'].apply(pd.Series)

df.head(5)
print('Number of tweets with positive sentiment:', len(df.loc[df['compound']>=0]))

print('Number of tweets with negative sentiment:', len(df.loc[df['compound']<0]))
from nltk.corpus import stopwords

stopwords = stopwords.words("english")



def clean_data(col):

    """Removes @mentions, <tags>, stopwords, urls, RTs, applies lower()"""

    allwords = ' '.join(col)

    tags_pattern = re.compile(r"<.*?>|(@[A-Za-z0-9_]+)")

    allwords = tags_pattern.sub('', allwords)

    allwords = re.sub(r'https\S+', '', allwords, flags=re.MULTILINE)

    allwords = allwords.replace('RT ', '').lower()

    allwords = ' '.join([word for word in allwords.split() if word not in stopwords])



    return allwords
allwords = clean_data(df.text)

cloud = wordcloud.WordCloud(background_color='white',

                            colormap='Blues',

                            max_font_size=200,

                            width=1000,

                            height=500,

                            max_words=300,

                            relative_scaling=0.5,

                            collocations=False).generate(allwords)

plt.figure(figsize=(20,15))

plt.imshow(cloud, interpolation="bilinear")
positive_df = df.loc[df['compound']>=0]

allwords = clean_data(positive_df.text)

cloud = wordcloud.WordCloud(background_color='white',

                            colormap='Greens',

                            max_font_size=200,

                            width=1000,

                            height=500,

                            max_words=300,

                            relative_scaling=0.5,

                            collocations=True).generate(allwords)

plt.figure(figsize=(20,15))

plt.imshow(cloud, interpolation="bilinear")
negative_df = df.loc[df['compound']<0]

allwords = clean_data(negative_df.text)

cloud = wordcloud.WordCloud(background_color='white',

                            colormap='Reds',

                            max_font_size=200,

                            width=1000,

                            height=500,

                            max_words=300,

                            relative_scaling=0.5,

                            collocations=True).generate(allwords)

plt.figure(figsize=(20,15))

plt.imshow(cloud, interpolation="bilinear")