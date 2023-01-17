# Importing the required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from textblob import TextBlob  

from bs4 import BeautifulSoup

import re

from wordcloud import WordCloud, STOPWORDS

import networkx as nx          

import nltk

from nltk.corpus import stopwords

import itertools

import collections

from nltk import bigrams

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import WordNetLemmatizer

from sklearn.cluster import KMeans

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.util import *



import os

for dirname, _, filenames in os.walk('../input/demonetization-tweets.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
tweets = pd.read_csv('../input/demonetization-tweets.csv', encoding = "ISO-8859-1")

tweets.head()
tweets.shape
display(tweets.text.head(10))
print(tweets['retweetCount'])
def clean(x):

    #Remove Html  

    x=BeautifulSoup(x).get_text()

    

    #Remove Non-Letters

    x=re.sub('[^a-zA-Z]',' ',x)

    

    #Convert to lower_case and split

    x=x.lower().split()

    

    #Remove stopwords

    stop=set(stopwords.words('english'))

    words=[w for w in x if not w in stop]

    

    #join the words back into one string

    return(' '.join(words))

tweets['text']=tweets['text'].apply(lambda x:clean(x))
# tweets.head()

display(tweets.text.head(10))
words_in_tweet = [tweets.lower().split() for tweets in tweets.text]

words_in_tweet[0]
stop_words = set(stopwords.words('english'))



# View a few words from the set

list(stop_words)[0:10]

tweets_nsw = [[word for word in tweet_words if not word in stop_words]

              for tweet_words in words_in_tweet]

tweets_nsw[0]
all_words_nsw = list(itertools.chain(*tweets_nsw))



counts_nsw = collections.Counter(all_words_nsw)



counts_nsw.most_common(15)
clean_tweets_nsw = pd.DataFrame(counts_nsw.most_common(15),

                             columns=['words', 'count'])



fig, ax = plt.subplots(figsize=(8, 8))



# Plot horizontal bar graph

clean_tweets_nsw.sort_values(by='count').plot.barh(x='words',

                      y='count',

                      ax=ax,

                      color="green")



ax.set_title("Common Words Found in Tweets (Without Stop Words)")



plt.show()
collection_words = ['rt', 'co', 'http', 'https', 'j', 'k']

tweets_nsw_nc = [[w for w in word if not w in collection_words]

                 for word in tweets_nsw]

tweets_nsw_nc[0]
terms_bigram = [list(bigrams(tweet)) for tweet in tweets_nsw_nc]

terms_bigram[0]
# Flatten list of bigrams in clean tweets

bigrams = list(itertools.chain(*terms_bigram))



# Create counter of words in clean bigrams

bigram_counts = collections.Counter(bigrams)



bigram_counts.most_common(20)
#Top 20 most common bigrams

bigram_df = pd.DataFrame(bigram_counts.most_common(20),

                             columns=['bigram', 'count'])

bigram_df
d = bigram_df.set_index('bigram').T.to_dict('records')

G = nx.Graph()

# Create connections between nodes

for k, v in d[0].items():

    G.add_edge(k[0], k[1], weight=(v * 3))



fig, ax = plt.subplots(figsize=(12, 10))



pos = nx.spring_layout(G, k=4)



# Plot networks

nx.draw_networkx(G, pos,

                 font_size=11,

                 fontweight='bold',

                 width=2,

                 edge_color='grey',

                 node_color='green',

#                  edge_length = 10,

                 with_labels = False,

                 ax=ax)



# Create offset labels

for key, value in pos.items():

    x, y = value[0]+.00167, value[1]+.045

    ax.text(x, y,

            s=key,

            bbox=dict(facecolor='orange', alpha=0.5),

            horizontalalignment='center', fontsize=10)

    

plt.show()
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.util import *

from nltk.corpus import stopwords

from nltk import tokenize



sid = SentimentIntensityAnalyzer()



tweets['sentiment_compound_polarity']=tweets.text.apply(lambda x:sid.polarity_scores(x)['compound'])

tweets['sentiment_neutral']=tweets.text.apply(lambda x:sid.polarity_scores(x)['neu'])

tweets['sentiment_negative']=tweets.text.apply(lambda x:sid.polarity_scores(x)['neg'])

tweets['sentiment_pos']=tweets.text.apply(lambda x:sid.polarity_scores(x)['pos'])

tweets['sentiment_type']=''

tweets.loc[tweets.sentiment_compound_polarity>0,'sentiment_type']='Positive'

tweets.loc[tweets.sentiment_compound_polarity==0,'sentiment_type']='Neutral'

tweets.loc[tweets.sentiment_compound_polarity<0,'sentiment_type']='Negative'

tweets.head(3)
tweets.sentiment_type.value_counts()
# fig = plt.figure()

# fig.savefig('Sentiment_bar_plot.pdf')

colors = ['green', 'yellow', 'red']

tweets.sentiment_type.value_counts().plot(kind='bar',figsize=(9, 7),edgecolor='k',title="Twitter Sentiment Analysis- Demonetisation (Bar Graph)", color=colors)

# plt.savefig('Sentiment_bar_plot.png', dpi=100)
colors = ['green', 'yellow', 'red']

explode = (0, 0.08, 0.1)

tweets.sentiment_type.value_counts().plot(kind='pie', figsize=(9, 7), title="Twitter Sentiment Analysis- Demonetisation (Pie Graph)", colors=colors, explode=explode,autopct='%1.1f%%', shadow=False)

# plt.savefig('Sentiment_pie_plot.png', dpi=100)
tweets['statusSource_new'] = ''

for i in range(len(tweets['statusSource'])):

    m = re.search('(?<=>)(.*)', tweets['statusSource'][i])

    try:

        tweets['statusSource_new'][i]=m.group(0)

    except AttributeError:

        tweets['statusSource_new'][i]=tweets['statusSource'][i] 

#print(tweets['statusSource_new'].head())   

tweets['statusSource_new'] = tweets['statusSource_new'].str.replace('</a>', ' ', case=False)

tweets['statusSource_new2'] = ''



for i in range(len(tweets['statusSource_new'])):

    if tweets['statusSource_new'][i] not in ['Twitter for Android ','Twitter Web Client ','Twitter for iPhone ']:

        tweets['statusSource_new2'][i] = 'Others'

    else:

        tweets['statusSource_new2'][i] = tweets['statusSource_new'][i] 

print(tweets['statusSource_new2']) 



tweets_by_type2 = tweets.groupby(['statusSource_new2'])['retweetCount'].sum()

tweets_by_type2.rename("",inplace=True)

explode = (0, 0, 0.09, 0)

tweets_by_type2.transpose().plot(kind='pie',figsize=(9, 7),autopct='%1.1f%%',shadow=False,explode=explode)

plt.legend(bbox_to_anchor=(1, 1), loc=6, borderaxespad=0.)

plt.title('Number of retweetcount by Source bis', bbox={'facecolor':'0.8', 'pad':5})
tweets['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in tweets.text]       

####

vectorizer = TfidfVectorizer(max_df=0.5,max_features=10000,min_df=10,stop_words='english',use_idf=True)

X = vectorizer.fit_transform(tweets['text_lem'].str.upper())

print("Shape of X is: ", X.shape, '\n')

print("Clusters are: ", '\n')

km = KMeans(n_clusters=100,init='k-means++',max_iter=200,n_init=1)

km.fit(X)

terms = vectorizer.get_feature_names()

order_centroids = km.cluster_centers_.argsort()[:,::-1]

for i in range(100):

    print("cluster %d:" %i, end='')

    for ind in order_centroids[i,:10]:

        print(' %s' % terms[ind], end='')

    print() 