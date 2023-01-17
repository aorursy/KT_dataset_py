import pandas as pd

import numpy as np

import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import gensim

from gensim.corpora import Dictionary

from gensim.models import TfidfModel

from sklearn.model_selection import train_test_split

import warnings

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import pyLDAvis

import pyLDAvis.gensim

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
%matplotlib inline

pd.set_option("display.max_colwidth", 300)

stp_words = stopwords.words('english')

stp_words.extend(['rt', 'vs', 'amp', 'quot', 'gt'])
#Read Given Tweets

tweet_data = pd.read_excel("../input/TWEET STACK.XLSX", dtype={'TweetFulltext' : str})

tweet_data.head(10)
def clean_tweets(tweet_txt):

    #Removing Handles

    tweet_txt_new = re.sub("@[\w]*", " ", tweet_txt)

    #Removing URLs

    tweet_txt_new = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", tweet_txt_new)

    #Removing punctation marks

    tweet_txt_new = re.sub("[^a-zA-Z#]", " ", tweet_txt_new)

    #Removing hashes

    tweet_txt_new = tweet_txt_new.replace('#', '')

    #Removing stop words

    words = [word.lower() for word in tweet_txt_new.split()]

    

    for stp_word in stp_words:

        if stp_word in words:

            words = list(filter(lambda a: a != stp_word, words))

    #Stemming words        

    stemmer = PorterStemmer()

    words_stemmed = [stemmer.stem(i) for i in words]

    return " ".join(words_stemmed)

tweet_data['TweetFulltext_cleaned'] = tweet_data.apply(lambda x: clean_tweets(x['TweetFulltext']), axis=1)

tweet_data.head(10)
tweet_data = tweet_data.drop('TweetFulltext', axis=1)

tweet_data.head(10)
# Generating word-cloud to see the most prominent words in the corpus

# The size of the word is directly proportional to its occurence.

combined_corpus_text = ' '.join([text for text in tweet_data['TweetFulltext_cleaned']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(combined_corpus_text) 

plt.figure(figsize=(25, 25))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
# Tokenize tweets

tweet_data_tokenized = tweet_data.TweetFulltext_cleaned.apply(lambda x: x.split())
#Creating index to token mapping dictionary

dct = Dictionary(tweet_data_tokenized)



#Creating term-frequency matrix

corpus = [dct.doc2bow(tweet) for tweet in tweet_data_tokenized]



#Creating TF-IDF model

model = TfidfModel(corpus)
LdaModel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, passes=10, id2word=dct)

LdaModel.print_topics()
#Calculate model perplexity. Lower the better

LdaModel.log_perplexity(corpus)
pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(LdaModel, corpus, dct)

vis