# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Imports statements.
import numpy as np
import matplotlib.pyplot as plt
import re
import multiprocessing
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from IPython.display import display
%matplotlib inline
def display_all_details(dataframe):
    print(('='*50)+'DATA'+('='*50))
    print(('-'*50)+'SHAPE'+('-'*50))
    print(dataframe.shape)
    print(('-'*50)+'COLUMNS'+('-'*50))
    print(dataframe.columns)
    print(('-'*50)+'DESCRIBE'+('-'*50))
    print(dataframe.describe())
    print(('-'*50)+'INFO'+('-'*50))
    print(dataframe.info())
    print(('='*50)+'===='+('='*50))
covid19_tweets_data = pd.read_csv('../input/covid19-tweets/covid19_tweets.csv')
covid19_tweets_data.head()
covid19_tweets_data.tail()
display_all_details(covid19_tweets_data)
covid19_tweets_data.isnull().sum()
def display_missing_values_info(df):
    missing_values_count_df = df.isnull().sum()
    print(('='*50)+'DATA WITH MISSING VALUES'+('='*50))
    print(missing_values_count_df[missing_values_count_df>0])
    print(('='*50)+'DATA WITHOUT MISSING VALUES'+('='*50))
    print(missing_values_count_df[missing_values_count_df==0])
    
    
display_missing_values_info(covid19_tweets_data)
for tweets in covid19_tweets_data.text.head(20):
    print(tweets)
for tweet in covid19_tweets_data.text:
    link = re.search("(?P<url>https?://[^\s]+)", myString).group("url")
    if link!=None:
        covid19_tweets_data['links'] = link
    else:
        covid19_tweets_data['links'] = pd.NA
        
covid19_tweets_data.head()
covid19_tweets_data.links.isna().sum()
def clean_text_column(row):
    text = row['text'].lower()
    text = re.sub("(?P<url>https?://[^\s]+)",'',text)
    text = re.sub(r'[^(a-zA-Z\s)]','',text)
    text = re.sub(r'\(','',text)
    text = re.sub(r'\)','',text)
    text = text.replace('\n',' ')
    text = text.strip()
    return text
covid19_tweets_data['cleaned_text'] = covid19_tweets_data.apply(clean_text_column,axis = 1)
for tweets in covid19_tweets_data.cleaned_text.head(20):
    print(tweets)
covid19_tweets_data.cleaned_text.str.isspace().sum()
covid19_tweets_data.shape
covid19_tweets_data.drop(covid19_tweets_data[covid19_tweets_data['cleaned_text'].str.isspace()==True].index,inplace = True)
covid19_tweets_data.shape
sent = [row for row in covid19_tweets_data.cleaned_text]
phrases = Phrases(sent, min_count=1, progress_per=50000)
bigram = Phraser(phrases)
sentences = bigram[sent]
sentences[:10]
filtered_sentences = []
for tweet in sentences:
    filtered_sentences.append(remove_stopwords(tweet))
filtered_sentences
filtered_sentences_2 = []
for tweet in filtered_sentences:
    filtered_sentences_2.append(re.sub(r'\b\w{1,2}\b', '',tweet))
filtered_sentences_2
w2v_model = Word2Vec(min_count=3,
                     window=4,
                     size=200,
                     sample=1e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=multiprocessing.cpu_count()-1)
w2v_model.build_vocab(filtered_sentences_2, progress_per=50000)
w2v_model.train(filtered_sentences_2, total_examples=w2v_model.corpus_count, epochs=100, report_delay=1)
w2v_model.init_sims(replace=True)

w2v_model.save("word2vec.model")
word_vectors = Word2Vec.load("./word2vec.model").wv

model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(X=word_vectors.vectors)
len(model.cluster_centers_[0])
word_vectors.similar_by_vector(model.cluster_centers_[0], topn=10, restrict_vocab=None)
word_vectors.similar_by_vector(model.cluster_centers_[1], topn=10, restrict_vocab=None)
positive_cluster_center = model.cluster_centers_[0]
negative_cluster_center = model.cluster_centers_[1]
words = pd.DataFrame(word_vectors.vocab.keys())
words.columns = ['words']

words['vectors'] = words.words.apply(lambda x: word_vectors.wv[f'{x}'])
words.vectors[0].dtype

words['cluster'] = words.vectors.apply(lambda x: model.predict(np.array([x])))
words.cluster

words.cluster = words.cluster.apply(lambda x: x[0])
words.cluster.unique()
words['cluster_value'] = [1 if i==0 else -1 for i in words.cluster]

words['closeness_score'] = words.apply(lambda x: 1/(model.transform([x.vectors]).min()), axis=1)

words['sentiment_coeff'] = words.closeness_score * words.cluster_value
words.head()
words[['words', 'sentiment_coeff']].to_csv('sentiment_dictionary.csv', index=False)
sentiment_map = pd.read_csv('./sentiment_dictionary.csv')
sentiment_dict = dict(zip(sentiment_map.words.values, sentiment_map.sentiment_coeff.values))
sentiment_dict
tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
tfidf.fit(covid19_tweets_data.cleaned_text)
features = pd.Series(tfidf.get_feature_names())
transformed = tfidf.transform(covid19_tweets_data.cleaned_text)
'covid' in features.unique()
def create_tfidf_dictionary(x, transformed_file, features):
    '''
    create dictionary for each input sentence x, where each word has assigned its tfidf score
    
    inspired  by function from this wonderful article: 
    https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34
    
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer

    '''
    vector_coo = transformed_file[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo

def replace_tfidf_words(x, transformed_file, features):
    '''
    replacing each word with it's calculated tfidf dictionary with scores of each word
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer
    '''
    dictionary = create_tfidf_dictionary(x, transformed_file, features)   
    return list(map(lambda y:dictionary[f'{y}'], x.cleaned_text.split()))
replaced_tfidf_scores = covid19_tweets_data.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)
def replace_sentiment_words(word, sentiment_dict):
    '''
    replacing each word with its associated sentiment score from sentiment dict
    '''
    try:
        out = sentiment_dict[word]
    except KeyError:
        out = 0
    return out
replaced_closeness_scores = covid19_tweets_data.cleaned_text.apply(lambda x: list(map(lambda y: replace_sentiment_words(y, sentiment_dict), x.split())))
covid19_tweets_data.columns
replacement_df = pd.DataFrame(data=[replaced_closeness_scores, replaced_tfidf_scores, covid19_tweets_data.cleaned_text]).T
replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'sentence']
replacement_df['sentiment_rate'] = replacement_df.apply(lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
replacement_df['prediction'] = (replacement_df.sentiment_rate>0).astype('int8')
#replacement_df['sentiment'] = [1 if i==1 else 0 for i in replacement_df.sentiment]
replacement_df.head()
replacement_df.prediction.value_counts()
