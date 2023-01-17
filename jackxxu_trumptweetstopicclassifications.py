import pandas as pd

import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

import numpy as np

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

np.random.seed(2018)

# import nltk

# nltk.download('wordnet')

import warnings

warnings.filterwarnings('ignore')



data = pd.read_csv('/kaggle/input/market-volatility/TrumpTweets.csv', error_bad_lines=False);

data_text = data[['text']]

data_text['index'] = data_text.index

documents = data_text
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))

STOPWORDS.add('tco') #additional manual word removal

STOPWORDS.add('https') #additional manual word removal



def clean_text(text):

    """

        text: a string

        

        return: modified initial string

    """

    text = BeautifulSoup(text, "lxml").text # HTML decoding

    text = text.lower() # lowercase text

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text

    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text

    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text

    return text



data_text['text'] = data_text['text'].apply(clean_text)
stemmer = SnowballStemmer('english')



def lemmatize_stemming(text):

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))



def preprocess(text):

    result = []

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:

            result.append(lemmatize_stemming(token))

    return result



processed_docs = documents['text'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
from gensim import corpora, models

from pprint import pprint

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=30, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model.print_topics(-1):

    print('Topic: {} \nWords: {}'.format(idx, topic))
data_text.head()
def topic(tweet_text): 

    bow_vector = dictionary.doc2bow(preprocess(tweet_text))

    topic_index = sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1])[0][0]

    return topic_index



for i in range(1, len(data_text)):

    data_text.loc[i, 'topic'] = topic(data_text.loc[i, 'text'])
tariff_df = data_text.query('topic == 23 or topic == 25')

print(tariff_df.shape)

tariff_df.head()