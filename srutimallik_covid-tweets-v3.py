import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) 

from nltk.tokenize import word_tokenize

from nltk.util import ngrams

import gensim

import re

from collections import Counter, defaultdict

from wordcloud import WordCloud, STOPWORDS

RANDOM_SEED = 1
tweets = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        df = pd.read_csv(os.path.join(dirname, filename), index_col=None, header=0)

        df = df[(df['lang']=='en') & (df['country_code'] == 'US')]

        row, cols = df.shape

        date = [filename[0:10]]*row

        df['date_of_tweet'] = date

        tweets.append(df)

        

tweets_en_US = pd.concat(tweets, axis=0, ignore_index=True)

del tweets



# random samples of tweets 

print(tweets_en_US.text.sample(5))



# print columns of dataframe

# print(tweets_en_US.columns)



#drop columns with user sensitive information 

tweets_en_US_encrypted =tweets_en_US.copy()

tweets_en_US_encrypted.drop(['status_id','user_id','screen_name','source','reply_to_status_id',

                                    'reply_to_user_id','is_retweet','place_full_name','place_type',

                                    'reply_to_screen_name','is_quote','followers_count','friends_count',

                                    'account_lang','account_created_at','verified'],axis=1, inplace = True)

print(tweets_en_US_encrypted.columns)
tweets_len = tweets_en_US_encrypted['text'].str.split().apply(lambda x: len(x))



plt.figure(figsize = (6,6))

sns.distplot(tweets_len, bins = 20, kde = 'False',)

plt.xlabel('Length of Tweets')

plt.title('Distribution of length of Tweets')

plt.show()
tweets_en_US_encrypted.sort_values('date_of_tweet', inplace=True)



plt.figure(figsize=(12,8))

sns.countplot(y = tweets_en_US_encrypted['date_of_tweet'])

plt.show()
import string

punct = string.punctuation



import emoji



def deEmojify(text):

    allchars = [str for str in text]

    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]

    clean_text = ' '.join([str for str in text.split() 

                           if not any(i in str for i in emoji_list)])

    return clean_text



def removeURL(text):

    clean_text = re.sub(r"http\S+", "", text)

    return clean_text



def tokenize_tweets(dataframe):

    tokenized_data = []

    for i,tweet in enumerate(dataframe['text']):

        sentence = []

        tweet = deEmojify(tweet)

        tweet = removeURL(tweet)

        for w in tweet.split():

            if w.lower() not in stop_words and w not in punct and w!='&amp;':

                sentence.append(w.lower())

        dataframe['text'][i] = ' '.join(sentence)

        tokenized_data.append(sentence)       

    return tokenized_data, dataframe



tweets_US_tokenized, tweets_en_US_encrypted = tokenize_tweets(tweets_en_US_encrypted)

print(tweets_en_US_encrypted['text'][:5])

print(tweets_US_tokenized[:5])
from PIL import Image

def show_WordCloud(data_list, title = None):

    data_list_compiled = ''

    data_list_compiled += " ".join(data_list)+" "

    wordcloud = WordCloud(background_color = 'white', max_words = 200, min_font_size = 8, max_font_size=40).generate(str(data_list_compiled))

    

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.title(title)

    plt.show()

    

def top_list_elements(list_data, N = 20):

    """returns a dictionary of hashtags and the number of times they have been used"""

    count =Counter(list_data)

    top_elements = dict(sorted(count.items(), key = lambda x:x[1], reverse = True)[:N])

    return top_elements
def extract_Hashtags(tokenized_data):

    """ returns list of hashtags used """

    list_hashtag = []

    for tweets in tokenized_data:

        list_hashtag.append([w for w in tweets if w.startswith('#')])

    return [item for sublist in list_hashtag for item in sublist]



tweets_Hashtag = extract_Hashtags(tweets_US_tokenized)

top_N_Hashtags = top_list_elements(tweets_Hashtag, N=100)



plt.figure(figsize = (10, 6))

sns.barplot(x = list(top_N_Hashtags.values())[:20], y = list(top_N_Hashtags.keys())[:20])

plt.show()



plt.figure(figsize = (16, 16))

show_WordCloud(list(top_N_Hashtags.keys()))



def extract_Mentions(tokenized_data):

    """ returns list of mentions used """

    list_mentions = []

    for tweets in tokenized_data:

        list_mentions.append([w for w in tweets if w.startswith('@')])

    return [item for sublist in list_mentions for item in sublist]



tweets_mentions = extract_Mentions(tweets_US_tokenized)

top_N_Mentions = top_list_elements(tweets_mentions, N=50)



plt.figure(figsize = (10, 6))

sns.barplot(x = list(top_N_Mentions.values())[:20], y = list(top_N_Mentions.keys())[:20])

plt.show()



plt.figure(figsize = (16, 16))

show_WordCloud(list(top_N_Mentions.keys()))

from textblob import TextBlob



import warnings

warnings.filterwarnings("ignore")



tweets_en_US_encrypted['sentiment'] = ' '

tweets_en_US_encrypted['polarity'] = None

for i,tweets in enumerate(tweets_en_US_encrypted.text) :

    blob = TextBlob(tweets)

    tweets_en_US_encrypted['polarity'][i] = blob.sentiment.polarity

    if blob.sentiment.polarity > 0 :

        tweets_en_US_encrypted['sentiment'][i] = 'positive'

    elif blob.sentiment.polarity < 0 :

        tweets_en_US_encrypted['sentiment'][i] = 'negative'

    else :

        tweets_en_US_encrypted['sentiment'][i] = 'neutral'

tweets_en_US_encrypted.head()
plt.figure(figsize = (16,8))

sns.countplot(x = tweets_en_US_encrypted['date_of_tweet'], hue = 'sentiment', 

              data = tweets_en_US_encrypted, palette = 'cool', saturation = 0.5)

plt.xticks(Rotation = 45)

plt.show()
plt.figure(figsize = (10, 6))

sns.distplot(tweets_en_US_encrypted['polarity'], bins = 30)

plt.xlabel('Polarity',size = 15)

plt.ylabel('Frequency',size = 15)

plt.show()
pos = tweets_en_US_encrypted['text'][tweets_en_US_encrypted['sentiment'] == 'positive']

neutral = tweets_en_US_encrypted['text'][tweets_en_US_encrypted['sentiment'] == 'neutral']

neg = tweets_en_US_encrypted['text'][tweets_en_US_encrypted['sentiment'] == 'negative']



def extract_list(dataframe):

    data = []

    for tweet in dataframe:

        data.append(tweet.split())

    return data



pos_list = extract_list(pos)

neu_list = extract_list(neutral)

neg_list = extract_list(neg)
# Check the Hashtags in tweets of different sentiments



pos_Hashtag = extract_Hashtags(pos_list)

top_N_pos_Hashtags = top_list_elements(pos_Hashtag, N=100)

plt.figure(figsize = (10,16))

sns.barplot(x = list(top_N_pos_Hashtags.values())[49:], y = list(top_N_pos_Hashtags.keys())[49:], 

            palette = 'cool', saturation = 0.5)

plt.title('Top 51-100 Positive Hashtags')

plt.show()
neu_Hashtag = extract_Hashtags(neu_list)

top_N_neu_Hashtags = top_list_elements(neu_Hashtag, N=100)

plt.figure(figsize = (10,16))

sns.barplot(x = list(top_N_neu_Hashtags.values())[49:], y = list(top_N_neu_Hashtags.keys())[49:], 

            palette = 'cool', saturation = 0.5)

plt.title('Top 51-100 Neutral Hashtags')

plt.show()
neg_Hashtag = extract_Hashtags(neg_list)

top_N_neg_Hashtags = top_list_elements(neg_Hashtag, N=100)

plt.figure(figsize = (10,16))

sns.barplot(x = list(top_N_neg_Hashtags.values())[49:], y = list(top_N_neg_Hashtags.keys())[49:], 

            palette = 'cool', saturation = 0.5)

plt.title('Top 51-100 Negative Hashtags')

plt.show()
# Check the mentions in tweets of different sentiments

pos_mention = extract_Mentions(pos_list)

top_N_pos_Mentions = top_list_elements(pos_mention, N=200)

plt.figure(figsize = (16,10))

show_WordCloud(top_N_pos_Mentions , 'POSITIVE')
neu_mention = extract_Mentions(neu_list)

top_N_neu_Mentions = top_list_elements(neu_mention, N=200)

plt.figure(figsize = (16,10))

show_WordCloud(top_N_neu_Mentions , 'NEUTRAL')
neg_mention = extract_Mentions(neg_list)

top_N_neg_Mentions = top_list_elements(neg_mention, N=200)

plt.figure(figsize = (16,10))

show_WordCloud(top_N_neg_Mentions , 'NEGATIVE')
plt.figure(figsize = (10,10))

show_WordCloud(pos , 'POSITIVE')



plt.figure(figsize = (10,10))

show_WordCloud(neutral , 'NEUTRAL')



plt.figure(figsize = (10,10))

show_WordCloud(neg , 'NEGATIVE')
from gensim.models import Word2Vec



def train_w2v(tokenized_corpus):

    w2v_model = Word2Vec(min_count = 20, sample = 0.05, negative = 10)

    w2v_model.build_vocab(tokenized_corpus)

    w2v_model.train(tokenized_corpus, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

    print('Trained....')

    return w2v_model



w2v_model = train_w2v(tweets_US_tokenized)

from sklearn.manifold import TSNE

def tsne_plot(model, word_list):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in word_list:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(16, 16)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.axis('off')

    plt.show()
top_N_Hashtags = top_list_elements(extract_Hashtags(tweets_US_tokenized), N=200)

tsne_plot(w2v_model, list(top_N_Hashtags.keys()))
top_N_Mentions = top_list_elements(extract_Mentions(tweets_US_tokenized), N=200)

tsne_plot(w2v_model, list(top_N_Mentions.keys()))