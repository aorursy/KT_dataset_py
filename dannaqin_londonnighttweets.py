import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
tweets2017 = pd.read_csv('../input/LondonNight_170101_170131.csv')
tweets2015 = pd.read_csv('../input/LondonNight_150101_150131.csv')
tweets2015.head()
tweets2015.count()
tweets2017.count()
# convert the datetime
tweets2015['date'] = pd.to_datetime(tweets2015['date'])
# tweets2017['date'] = pd.to_datetime(tweets2017['date'])
# datetime convertion for 2017 creates OutOfBound error. Haven't figure out why.
tweets2015['hour'] = tweets2015['date'].apply(lambda x:x.hour)
# extract tweets from 18:00 to 6:00
tweets2015_night = tweets2015.loc[(tweets2015.hour < 7)|(tweets2015.hour > 17)]
tweets2015_night.hour.value_counts().plot(kind = 'bar')
from wordcloud import WordCloud, STOPWORDS

def TWTwordcloud(tweets):
    stopwords = set(STOPWORDS)
    stopwords.add("London")
    stopwords.add("night")
    stopwords.add("https")
    stopwords.add("look")
    stopwords.add("meet")

    wordcloud = WordCloud(
        background_color="white",
        stopwords=stopwords,
        width = 600,
        height = 400,
        random_state = 200).generate(str(tweets['text']))
    
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Tweets London Night")

TWTwordcloud(tweets2015) 
def TWTwordcloud(tweets):
    stopwords = set(STOPWORDS)
    stopwords.add("London")
    stopwords.add("night")
    stopwords.add("https")
    stopwords.add("look")
    stopwords.add("meet")
    stopwords.add("Jonathan")

    wordcloud = WordCloud(
        background_color="white",
        stopwords=stopwords,
        width = 600,
        height = 400,
        random_state = 200).generate(str(tweets['text']))
    
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Tweets London Night")

TWTwordcloud(tweets2015) 
TWTwordcloud(tweets2017)
tweets2015_night = tweets2015_night.set_index('date')
tweets2015_night.head()
tweets2015_volume = tweets2015_night.resample('10T').count()
tweets2015_volume.tail()
tweets2015_volume.plot(y = 'id')
plt.title('number of tweets by minute(at night)')
# just ignore the flat line befor Jan 2015, seems some err occurs when resampling
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

from nltk import tokenize

sid = SentimentIntensityAnalyzer()

tweets2015['sentiment_compound_polarity']=tweets2015.text.apply(lambda x:sid.polarity_scores(x)['compound'])
tweets2015['sentiment_neutral']=tweets2015.text.apply(lambda x:sid.polarity_scores(x)['neu'])
tweets2015['sentiment_negative']=tweets2015.text.apply(lambda x:sid.polarity_scores(x)['neg'])
tweets2015['sentiment_pos']=tweets2015.text.apply(lambda x:sid.polarity_scores(x)['pos'])
tweets2015['sentiment_type']=''
tweets2015.loc[tweets2015.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
tweets2015.loc[tweets2015.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
tweets2015.loc[tweets2015.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
tweets2015.head()
import matplotlib
matplotlib.style.use('ggplot')

tweets2015_sentiment = tweets2015.groupby(['sentiment_type'])['sentiment_neutral'].count()
tweets2015_sentiment.rename("",inplace=True)
explode = (0, 0, 1.0)
plt.subplot(221)
tweets2015_sentiment.transpose().plot(kind='barh',figsize=(10, 6))
plt.title('Sentiment Analysis 1', bbox={'facecolor':'0.8', 'pad':0})
plt.subplot(222)
tweets2015_sentiment.plot(kind='pie',figsize=(10, 6),autopct='%1.1f%%',shadow=True,explode=explode)
plt.legend(bbox_to_anchor=(1, 1), loc=3, borderaxespad=0.)
plt.title('Sentiment Analysis 2', bbox={'facecolor':'0.8', 'pad':0})
plt.show()
tweets2015['count'] = 1
tweets2015_filtered = tweets2015[['hour', 'sentiment_type', 'count']]
pivot_tweets2015 = tweets2015_filtered.pivot_table(tweets2015_filtered, index=["sentiment_type", "hour"], aggfunc=np.sum)
pivot_tweets2015.head()
sentiment_type = pivot_tweets2015.index.get_level_values(0).unique()
#f, ax = plt.subplots(2, 1, figsize=(8, 10))
plt.plot( xticks=list(range(0,24)))

for sentiment_type in sentiment_type:
    split = pivot_tweets2015.xs(sentiment_type)
    split["count"].plot( legend=True, label='' + str(sentiment_type))
plt.title("Evolution of tweets' sentiments by hour", bbox={'facecolor':'0.8', 'pad':0})  
# clustering algorithms 
# from http://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html

pd.options.mode.chained_assignment = None
# nltk for nlp
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
# list of stopwords like articles, preposition
stop = set(stopwords.words('english'))
from string import punctuation
from collections import Counter

def tokenizer(text):
    try:
        tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text)]
        
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent

        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        tokens = list(filter(lambda t: t not in punctuation, tokens))
        tokens = list(filter(lambda t: t not in [u"'s", u"n't", u"...", u"''", u'``', u'amp', u'https',
                                                u'via', u"'re"], tokens))
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)

        filtered_tokens = list(map(lambda token: token.lower(), filtered_tokens))

        return filtered_tokens
    except Error as e:
        print(e)
tweets2015['tokens'] = tweets2015['text'].map(tokenizer)
for text, tokens in zip(tweets2015['text'].head(5), tweets2015['tokens'].head(5)):
    print('full text:', text)
    print('tokens:', tokens)
    print() 
from sklearn.feature_extraction.text import TfidfVectorizer

# min_df is minimum number of documents that contain a term t
# max_features is maximum number of unique tokens (across documents) that we'd consider

vectorizer = TfidfVectorizer(min_df=5, max_features=10000,tokenizer=tokenizer, ngram_range=(1, 2))
vz = vectorizer.fit_transform(list(tweets2015['text']))
from sklearn.cluster import MiniBatchKMeans

num_clusters = 10
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, 
                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
kmeans = kmeans_model.fit(vz)
kmeans_clusters = kmeans.predict(vz)
kmeans_distances = kmeans.transform(vz)
sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(num_clusters):
    print("Cluster %d:" % i)
    aux = ''
    for j in sorted_centroids[i, :10]:
        aux += terms[j] + ' | '
    print(aux)
    print() 
