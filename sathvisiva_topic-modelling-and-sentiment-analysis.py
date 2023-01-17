# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import nltk

from nltk import wordpunct_tokenize

from nltk.corpus import stopwords

from wordcloud import WordCloud,STOPWORDS

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation

import string



import textblob as tb

from tqdm import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
news_feed = pd.read_csv('../input/news-week-aug24.csv', dtype={'publish_time': object})



news_feed['publish_hour'] = news_feed.publish_time.str[:10]

news_feed['publish_date'] = news_feed.publish_time.str[:8]

news_feed['publish_hour_only'] = news_feed.publish_time.str[8:10]

news_feed['publish_time_only'] = news_feed.publish_time.str[8:12]

days=news_feed['publish_date'].unique().tolist()



news_feed['dt_time'] = pd.to_datetime(news_feed['publish_time'], format='%Y%m%d%H%M')

news_feed['dt_hour'] = pd.to_datetime(news_feed['publish_hour'], format='%Y%m%d%H')

news_feed['dt_date'] = pd.to_datetime(news_feed['publish_date'], format='%Y%m%d')
news_feed.head()
feed_count = news_feed['feed_code'].value_counts()

feed_count = feed_count[:10,]

plt.figure(figsize=(10,5))

sns.barplot(feed_count.index , feed_count.values, alpha = 0.8)

plt.title("Top 10 feed")

plt.ylabel('No of Occurances', fontsize = 12)

plt.xlabel('feed code', fontsize = 12)

plt.xticks(rotation=70)

plt.show()
news_feed = news_feed.dropna()

news_feed.count()
englishStopWords = set(nltk.corpus.stopwords.words('english'))

nonEnglishStopWords = set(nltk.corpus.stopwords.words()) - englishStopWords

stopWordsDictionary = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}

news_feed.headline_text.dropna()
def get_language(text):

    if type(text) is str:

        text = text.lower()

    words = set(nltk.wordpunct_tokenize(text))

    return max(((lang, len(words & stopwords)) for lang, stopwords in stopWordsDictionary.items()), key = lambda x: x[1])[0]
news_feed['language'] = news_feed['headline_text'].apply(get_language)
language_count = news_feed['language'].value_counts()

language_count = language_count[:10]

plt.figure(figsize = (10,5))

sns.barplot(language_count.index, language_count.values, alpha = 0.8)

plt.title("Top 10 languages feed")

plt.ylabel('No of Occurances', fontsize = 12)

plt.xlabel('Language', fontsize = 12)

plt.xticks(rotation=70)

plt.show()
news_feed_english_df = news_feed[news_feed['language'] == 'english']

news_feed_english = news_feed_english_df['headline_text']



def showWordCloud(data):

    words = ' '.join(data)

    cleaned_word = " ".join([word for word in words.split()])

    wordcloud = WordCloud(stopwords = STOPWORDS,

                         background_color = 'black',

                         width = 2500,

                         height = 2500

                         ).generate(cleaned_word)

    plt.figure(1,figsize = (13,13))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()



showWordCloud(news_feed_english)
def display_topics(model, feature_names, no_top_words):

    for topic_idx , topic in enumerate(model.components_):

        print("Topic %d:" % (topic_idx))

        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words -1:-1]]))
no_features = 1000

tfidf_vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 2,max_features=no_features, stop_words = 'english')

tfidf = tfidf_vectorizer.fit_transform(news_feed_english)

tfidf_feature_names = tfidf_vectorizer.get_feature_names()

tf_vectorizer = CountVectorizer(max_df = 0.95, min_df = 2, max_features=no_features, stop_words='english')

tf = tf_vectorizer.fit_transform(news_feed_english)

tf_feature_names = tf_vectorizer.get_feature_names()
no_topic = 5
nmf = NMF(n_components=no_topic, random_state = 1, alpha =.1, l1_ratio=.5, init = 'nndsvd').fit(tfidf)
lda = LatentDirichletAllocation(n_topics=no_topic, max_iter = 5, learning_method = 'online', learning_offset=50., random_state=0).fit(tf)
no_top_words = 10

display_topics(nmf ,tfidf_feature_names, no_top_words)

display_topics(lda , tf_feature_names , no_top_words)
def sent(x):

    t = tb.TextBlob(x)

    return t.sentiment.polarity, t.sentiment.subjectivity
tqdm.pandas(leave = False, mininterval = 25)

vals = news_feed_english_df.headline_text.progress_apply(sent)
news_feed_english_df['polarity'] = vals.str[0]

news_feed_english_df['sub'] = vals.str[1]
def plot_data(df , col):

    mean_pol = list(dict(df.groupby(col)['polarity'].mean()).items())

    mean_pol.sort(key=lambda x: x[0])



    plt.subplots(figsize=(15, 10))

    plt.subplot(2, 2, 1)

    plt.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])

    plt.xticks(rotation=70)

    plt.title('Mean polarity over time')



    plt.subplot(2, 2, 2)

    mean_pol = list(dict(df.groupby(col)['sub'].mean()).items())

    mean_pol.sort(key=lambda x: x[0])

    plt.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])

    plt.xticks(rotation=70)

    plt.title('Mean subjectivity over time')



    plt.subplot(2, 2, 3)

    mean_pol = list(dict(df.groupby(col)['polarity'].std()).items())

    mean_pol.sort(key=lambda x: x[0])

    plt.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])

    plt.xticks(rotation=70)

    plt.title('Std Dev of polarity over time')



    plt.subplot(2, 2, 4)

    mean_pol = list(dict(df.groupby(col)['sub'].std()).items())

    mean_pol.sort(key=lambda x: x[0])

    plt.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])

    plt.xticks(rotation=70)

    plt.title('Std dev of subjectivity over time')
plot_data(news_feed_english_df , 'dt_hour')
plot_data(news_feed_english_df , 'dt_time')