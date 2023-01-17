import numpy as np

import pandas as pd

pd.set_option('display.max_colwidth', -1)



import plotly

import plotly.offline as pyo

import plotly.graph_objects as go



import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize





import re

import glob

import json

import scipy.sparse

import pickle

from pprint import pprint





from sklearn.feature_extraction.text import CountVectorizer





# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel

from gensim import matutils, models





# spacy for lemmatization

import spacy



# Plotting tools

import pyLDAvis

import pyLDAvis.gensim  # don't skip this

import matplotlib.pyplot as plt

from IPython.display import IFrame, HTML

%matplotlib inline



# Enable logging for gensim - optional

import logging



import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)

include_terms = ['sars-cov-2', 'covid-19', '2019-ncov', 'ncov-2019', 'betacoronavirus', 'pandemic', 'wuhan', 'virulence', 

                 'incidence', 'prevalence', 'mortality', 'self-resolve', 'immunity', 'community', 'spread' 'olfactory', 

                 'dysfunction', 'osmia', 'respiratory', 'pneumonia']
# load the data



metadata_path = '../input/CORD-19-research-challenge/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})



# get only the abstracts from the dataframe



abstracts = dict(title = [], abstract = [])



for ind, row in meta_df.iterrows():

    if pd.notna(row['abstract']) and pd.notna(row['title']):

        short = row['abstract'][0:250]

        t = row['title'][0:100]

        abstracts['title'].append(t)

        abstracts['abstract'].append(short)
pd.DataFrame(abstracts).rename(columns={'title':'Title', 'abstract':'Abstract'}).head()
def lower(data):

    data = data.lower()

    return data





def remove_punctuation(data):

    symbols = "!\"#$%&()*+./:;<=>?@[\]^_`{|}~\n"

    for i in symbols:

        data = data.replace(i, " ")

    #replace the comma seperately

    data = data.replace(',', ' ')

    return data



def remove_apostraphe(data):

    data = data.replace("'", "")

    return data



def remove_numbers(data):

    nums = "0123456789"

    for i in nums:

        data = data.replace(i, "")

        return data



def remove_stop_words(data):

    stop_words = stopwords.words('english')

    words = word_tokenize(str(data))

    new_text = ""

    for w in words:

        if w not in stop_words and len(w) > 1:

            new_text = new_text + " " + w

    return new_text
# clean and replace with cleaned words

tf_dict = {}

idf_count = {}

for i, ab in meta_df['abstract'].items():

    if pd.notna(ab):

        sent_dict = {}

        working = ab

        working = lower(working)

        working = remove_stop_words(working)

        working = remove_punctuation(working)

        working = remove_numbers(working)

        working = remove_apostraphe(working)

        working = remove_stop_words(working)

        tokens = working.split(' ')

    

        for word in tokens:

            for inc in include_terms:

                if inc in word:

                    if word in sent_dict.keys():

                        sent_dict[word] = sent_dict[word]+1

                    else:

                        sent_dict[word] = 1

                        if word in idf_count.keys():

                            idf_count[word] = idf_count[word]+1

                        else:

                            idf_count[word] = 1

        tf_dict[i] = sent_dict
# calculate the tf-idf matrix

index = list(tf_dict.keys())

columns = list(idf_count.keys())

print("There were {} keywords generated from the Included Terms list".format(len(columns)))



#create an empty array for the tf-idf matrix

data = np.zeros((len(index), len(columns)))



#populate the tf-idf matrix

for i, doc in enumerate(index):

    for j, word in enumerate(columns):

        if word in tf_dict[doc]:

            if idf_count[word] !=1:

                data[i][j] = tf_dict[doc][word] * np.log(len(index)/idf_count[word])

            else:

                data[i][j] = np.nan

        else:

            data[i][j] = np.nan

            

#turn the tf_idf matrix into a dataframe

df_tfidf = pd.DataFrame(data, index = index, columns = columns)
df_tfidf.iloc[:10, :8]
cs = df_tfidf['community-acquired']

top_cs = cs.sort_values(ascending=False)[0:10]

final_cs = meta_df['abstract'].iloc[top_cs.index].tolist()

final_cs_title = meta_df['title'].iloc[top_cs.index].tolist()

final_cs_abstra = []

for abstra in final_cs:

    short = abstra[0:250]

    final_cs_abstra.append(short)

    

pd.DataFrame({'Title': final_cs_title, 'Abstract': final_cs_abstra})
# ingest body text for articles 

df_covid = pd.read_csv('/kaggle/input/subsetlda2/subset1000.csv') # Articles combined. 

text = df_covid.drop(["paper_id","doi","title_abstract_body","Unnamed: 0", "abstract", "title"], axis=1) # drop all columns except body_text

words = []

for ii in range(0,len(text)):

    words.append(str(text.iloc[ii]['body_text']).split(" "))

    

# Build the bigram and trigram models

bigram = gensim.models.Phrases(words, min_count=5, threshold=100) # higher threshold fewer phrases.

trigram = gensim.models.Phrases(bigram[words], threshold=100)  



# Faster way to get a sentence clubbed as a trigram/bigram

bigram_mod = gensim.models.phrases.Phraser(bigram)

trigram_mod = gensim.models.phrases.Phraser(trigram)
# Define functions for stopwords, bigrams, trigrams and lemmatization

def remove_stopwords(texts):

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):

    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):

    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    """https://spacy.io/api/annotation"""

    texts_out = []

    for sent in texts:

        doc = nlp(" ".join(sent)) 

        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out
# NLTK Stop words

from nltk.corpus import stopwords

stop_words = stopwords.words('english') + stopwords.words('spanish') + stopwords.words('french')

stop_words.extend(['doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI'])

# Remove Stop Words

data_words_nostops = remove_stopwords(words)

# Form Bigrams

data_words_bigrams = make_bigrams(data_words_nostops)



# Initialize spacy 'en' model, keeping only tagger component (for efficiency)

nlp = spacy.load('en', disable=['parser', 'ner'])

nlp.max_length = 1900000 # increased for size of body texgt 

# Do lemmatization keeping only noun, adj, vb, adv

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
# Create Dictionary

id2word = corpora.Dictionary(data_lemmatized)



# Create Corpus

texts = data_lemmatized



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]
# Build LDA model with 20 Topics (Reduced Number of Topics for Kaggle)



lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=20, 

                                           update_every=1,

                                           chunksize=8000,

                                           passes=4,

                                           iterations=400, 

                                           eval_every=None,

                                           alpha='auto',

                                           per_word_topics=True)
# Print the Keyword in the 20 topics

from pprint import pprint



pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]
from IPython.display import Image

Image('/kaggle/input/ldastaticimages/lda_model-img.jpg', width = 1050)
from IPython.display import Image

Image(filename='/kaggle/input/staticimages/coherences_numTopics.jpg', width = 850) 
queries = pd.read_csv('../input/covid19-processed-data/queries.csv')



queries.rename(columns={'queries':'Queries'})
use_and_bm25 = pd.read_csv('../input/covid19-processed-data/use_and_bm25.csv')

top5 = use_and_bm25[use_and_bm25['task_id'] == 2].head(5)[['task_text_x','excerpt']]



top5.rename(columns={'task_text_x':'Query', 'excerpt':'Excerpt'})
question_answer_summary = pd.read_csv('../input/covid19-processed-data/question_answer_dataframe.csv',

                                      usecols=['RISK_FACTOR_QUESTION',

                                               'ANSWER_SUMMARY',

                                               'TOP_5_ARTICLES'])



qa_summary = question_answer_summary.rename(columns={'RISK_FACTOR_QUESTION':'Query',

                                        'ANSWER_SUMMARY':'Summary',

                                        'TOP_5_ARTICLES':'Top 5 Articles'})

qa_summary[['Query','Summary', "Top 5 Articles"]]