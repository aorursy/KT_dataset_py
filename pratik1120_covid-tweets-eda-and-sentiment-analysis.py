import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

from wordcloud import WordCloud, ImageColorGenerator

from plotly.offline import iplot

import warnings

warnings.filterwarnings("ignore")



import nltk

import re

import string

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")

pd.options.plotting.backend = "plotly"



data = pd.read_csv('../input/covid19-tweets/covid19_tweets.csv')
text = ",".join(review for review in data.text if 'COVID' not in review and 'https' not in review and 'Covid' not in review)

wordcloud = WordCloud(max_words=200, colormap='Set3',background_color="black").generate(text)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
data.head(3)
print('How many posts are made with #covid19? -> {}\n'.format(data.shape[0]))

print('How many unique users have posted? -> {}\n'.format(data.user_name.nunique()))

print('How many unique locations were the posts made from? -> {}\n'.format(data.user_location.nunique()))

print('How many users have more than 1 million followers(higher chances of spread)? -> {}\n'.format(data[data['user_followers']>1000000].user_name.nunique()))

print('How many users are verified(denoting a known person)? -> {}\n'.format(data[data['user_verified']==True].user_name.nunique()))

print('How many tweets are re-tweets? -> {}'.format(data[data['is_retweet']==True].shape[0]))
data.describe()
sns.heatmap(data.drop('is_retweet', axis=1).corr())

plt.title('Correlation in data')

plt.show()
fig = data.isnull().sum().reset_index().plot(kind='bar', x=0, y='index', color=0)

fig.update_layout(title='Mising Values Plot', xaxis_title='Count', yaxis_title='Column Names')

fig.show()
fig = px.box(data, y="user_followers", color="user_verified",

                   title="User Followers Distribution")

fig.show()
sns.FacetGrid(data, hue="user_verified", height=6,).map(sns.kdeplot, "user_followers" ,shade=True).add_legend()

plt.title('User Follower kdeplot')

plt.show()
sns.FacetGrid(data, hue="user_verified", height=6,).map(sns.kdeplot, "user_friends" ,shade=True).add_legend()

plt.title('User Friends kdeplot')

plt.show()
sns.FacetGrid(data, hue="user_verified", height=6,).map(sns.kdeplot, "user_favourites" ,shade=True).add_legend()

plt.title('User Favourites kdeplot')

plt.show()
fig = data.source.value_counts().reset_index().head(10).plot(kind='bar',x='index',y='source',color='source')

fig.update_layout(title='Top 10 sources of tweets', xaxis_title='Sources', yaxis_title='')

fig.show()
data['text_length'] = data['text'].str.len()

fig = px.violin(data, y="text_length", color="user_verified",

                   title="Text Length Distribution")

fig.show()
fig = data.user_location.value_counts().reset_index().head(10).plot(kind='bar',x='index',y='user_location',color='user_location')

fig.update_layout(title='Top 10 location of tweets', xaxis_title='Locations', yaxis_title='')

fig.show()
def clean_text(text):

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



#Source: https://www.kaggle.com/tamilsel/exploring-covid-19-tweets-and-sentiment-analysis



def text_preprocessing(text):

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    combined_text = ' '.join(tokenized_text)

    return combined_text



data['text'] = data['text'].apply(text_preprocessing)
data['hashtag_count'] = data['hashtags'].str.split(',').str.len()

data['hashtag_count'] = data['hashtag_count'].fillna(0.0)

fig = data.hashtag_count.value_counts().reset_index().head(7).plot(kind='bar', x='index', y='hashtag_count', color='hashtag_count')

fig.update_layout(title='Hashtag Count Distribution', xaxis_title='Hashtag Counts', yaxis_title='')

fig.show()
fig = data['text'].str.split().str.len().plot(kind='hist')

fig.update_layout(title='Word Count Distribution', xaxis_title='Word Count', yaxis_title='')

fig.show()
def get_top_n_words(corpus, n=None):

    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



common_words = get_top_n_words(data['text'], 15)

    

df1 = pd.DataFrame(common_words, columns = ['text' , 'count'])

fig = df1.plot(kind='bar', x='text', y='count', color='count')

fig.update_layout(yaxis_title='Count', title='Top 15 words before removing stop words')

fig.show()
#Source: https://www.kdnuggets.com/2019/05/complete-exploratory-data-analysis-visualization-text-data.html



def get_top_n_words(corpus, n=None):

    vec = CountVectorizer(stop_words = 'english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



common_words = get_top_n_words(data['text'], 15)

    

df1 = pd.DataFrame(common_words, columns = ['text' , 'count'])

fig = df1.plot(kind='bar', x='text', y='count', color='count')

fig.update_layout(yaxis_title='Count', title='Top 15 words after removing stop words')

fig.show()
def get_top_n_bigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



common_words = get_top_n_bigram(data['text'], 20)



df1 = pd.DataFrame(common_words, columns = ['text' , 'count'])

fig = df1.plot(kind='bar', y='text', x='count', color='count')

fig.update_layout(yaxis_title='Count', title='Top 20 bigrams before removing stop words')

fig.show()
def get_top_n_bigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



common_words = get_top_n_bigram(data['text'], 20)



df1 = pd.DataFrame(common_words, columns = ['text' , 'count'])

fig = df1.plot(kind='bar', y='text', x='count', color='count')

fig.update_layout(yaxis_title='Count', title='Top 20 bigrams after removing stop words')

fig.show()
def get_top_n_trigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



common_words = get_top_n_trigram(data['text'], 15)



df1 = pd.DataFrame(common_words, columns = ['text' , 'count'])

fig = df1.plot(kind='bar', y='text', x='count', color='count')

fig.update_layout(yaxis_title='Count', title='Top 15 trigrams before removing stop words')

fig.show()
def get_top_n_trigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



common_words = get_top_n_trigram(data['text'], 15)



df1 = pd.DataFrame(common_words, columns = ['text' , 'count'])

fig = df1.plot(kind='bar', y='text', x='count', color='count')

fig.update_layout(yaxis_title='Count', title='Top 15 trigrams after removing stop words')

fig.show()
model = SentimentIntensityAnalyzer()



def sentiment_score(txt):

    return model.polarity_scores(txt)['compound']



data["sentiment_score"] = data["text"].apply(sentiment_score)
fig = px.violin(data, y="sentiment_score", color="user_verified",

                   title="Sentiment Score Distribution")

fig.show()
df = data[data['sentiment_score']>0.5]



fig = df['user_location'].value_counts().reset_index().head(10).plot(kind='bar', y='user_location', x='index', color='user_location')

fig.update_layout(title='Location of most positive tweets', xaxis_title='Location', yaxis_title='')

fig.show()
df = data[data['sentiment_score']<0.5]



fig = df['user_location'].value_counts().reset_index().head(10).plot(kind='bar', y='user_location', x='index', color='user_location')

fig.update_layout(title='Location of most negative tweets', xaxis_title='Location', yaxis_title='')

fig.show()
data['date'] = pd.to_datetime(data['date'])

data['day'] = data['date'].dt.day



df = data[['day','sentiment_score']].copy()

df['avg_sentiment'] = df.groupby('day')['sentiment_score'].transform('mean')

df.drop('sentiment_score',axis=1,inplace=True)

df = df.drop_duplicates().sort_values('day')



df.plot(x='day', y='avg_sentiment', title='Sentiment of posts vs days in a month')
fig = px.scatter(data[data['user_followers']<20000000], x='sentiment_score', y='user_followers', color='user_followers')

fig.update_layout(title='Sentiment_score vs User_followers')

fig.show()