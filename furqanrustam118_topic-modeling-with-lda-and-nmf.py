import pandas as pd

import io



df = pd.read_csv('../input/all-source-metadata-202013csv/all_sources_metadata_2020-03-13.csv')

df = df[['title','abstract']]

df1=df.dropna()

x=df1.reset_index()

x
import re

import numpy as np

import pandas as pd

from pprint import pprint



# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel



# spacy for lemmatization

import spacy

import re

import nltk

from nltk.corpus import stopwords

stop_words=set(stopwords.words('english'))





# Plotting tools

import pyLDAvis

import pyLDAvis.gensim  # don't skip this

import matplotlib.pyplot as plt

%matplotlib inline



# Enable logging for gensim - optional

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)



import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



data_words = list(sent_to_words(x.abstract))
# Build the bigram and trigram models

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  



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
# Remove Stop Words

import re

import nltk

from nltk.corpus import stopwords

stop_words=set(stopwords.words('english'))

data_words_nostops = remove_stopwords(data_words)



# Form Bigrams

data_words_bigrams = make_bigrams(data_words_nostops)



# Initialize spacy 'en' model, keeping only tagger component (for efficiency)

# python3 -m spacy download en

nlp = spacy.load('en', disable=['parser', 'ner'])



# Do lemmatization keeping only noun, adj, vb, adv

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
# Create Dictionary

id2word = corpora.Dictionary(data_lemmatized)



# Create Corpus

texts = data_lemmatized



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=10, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=100,

                                           passes=10,

                                           alpha='auto',

                                           per_word_topics=True)
# Print the Keyword in the 10 topics

pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]
# Compute Perplexity

print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.



# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
import pandas as pd;

import numpy as np;

import scipy as sp;

import sklearn;

import sys;

from nltk.corpus import stopwords;

import nltk;

from gensim.models import ldamodel

import gensim.corpora;

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;

from sklearn.decomposition import NMF;

from sklearn.preprocessing import normalize;

import pickle;
data_text = x[['abstract']];
data_text = data_text.astype('str');

for idx in range(len(data_text)):

    

    #go through each word in each data_text row, remove stopwords, and set them on the index.

    data_text.iloc[idx]['abstract'] = [word for word in data_text.iloc[idx]['abstract'].split(' ') if word not in stopwords.words()];

    

    #print logs to monitor output

    if idx % 1000 == 0:

        sys.stdout.write('\rc = ' + str(idx) + ' / ' + str(len(data_text)));#save data because it takes very long to remove stop words

pickle.dump(data_text, open('data_text.dat', 'wb'))#get the words as an array for lda input

train_headlines = [value[0] for value in data_text.iloc[0:].values];
vectorizer = CountVectorizer(analyzer='word', max_features=5000);

x_counts = vectorizer.fit_transform(train_headlines_sentences);
transformer = TfidfTransformer(smooth_idf=False);

x_tfidf = transformer.fit_transform(x_counts);
xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
num_topics = 10;

#obtain a NMF model.

model = NMF(n_components=num_topics, init='nndsvd');#fit the model

model.fit(xtfidf_norm)
def get_nmf_topics(model, n_top_words):

    

    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.

    feat_names = vectorizer.get_feature_names()

    

    word_dict = {};

    for i in range(num_topics):

        

        #for each topic, obtain the largest values, and add the words they map to into the dictionary.

        words_ids = model.components_[i].argsort()[:-20 - 1:-1]

        words = [feat_names[key] for key in words_ids]

        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;

    

    return pd.DataFrame(word_dict);
get_nmf_topics(model, 20)