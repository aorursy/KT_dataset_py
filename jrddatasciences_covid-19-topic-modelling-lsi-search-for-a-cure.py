#Upgrade pip and install tqdm

!pip install --upgrade pip

!pip install tqdm

!pip install pyLDAvis
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.3/en_core_sci_md-0.2.3.tar.gz
#import packages

import json

from pprint import pprint

import numpy as np

import pandas as pd

import os

import zipfile

from tqdm import tqdm

from copy import deepcopy

import spacy

import en_core_sci_md

from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import PCA, LatentDirichletAllocation

from scipy.spatial.distance import jensenshannon

import joblib

import matplotlib.pyplot as plt

from IPython.display import HTML, display

from ipywidgets import interact, Layout, HBox, VBox, Box

import ipywidgets as widgets

from IPython.display import clear_output

import pyLDAvis

import pyLDAvis.sklearn

pyLDAvis.enable_notebook()

plt.style.use("dark_background")
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df= pd.read_csv('/kaggle/input/all_papers_filtered.csv')

df.shape
df.head()
all_texts = df.text
# example snippet

all_texts[42][0:500]
# medium model

nlp = en_core_sci_md.load(disable=["tagger", "parser", "ner"])

nlp.max_length = 1700000
!python -m spacy download fr_core_news_sm

!python -m spacy download es_core_news_sm

!python -m spacy download it_core_news_sm

!python -m spacy download de_core_news_sm

!python -m spacy download en_core_web_sm

!python -m spacy download xx_ent_wiki_sm
import fr_core_news_sm

import es_core_news_sm

import it_core_news_sm

import de_core_news_sm

import en_core_web_sm

import xx_ent_wiki_sm



nlp_fr = fr_core_news_sm.load()

nlp_es = es_core_news_sm.load()

nlp_it = it_core_news_sm.load()

nlp_de = de_core_news_sm.load()

nlp_en = en_core_web_sm.load()

nlp_xx = xx_ent_wiki_sm.load()



spacy_stopwords_fr = list(spacy.lang.fr.stop_words.STOP_WORDS)

spacy_stopwords_es = list(spacy.lang.es.stop_words.STOP_WORDS)

spacy_stopwords_it = list(spacy.lang.it.stop_words.STOP_WORDS)

spacy_stopwords_de = list(spacy.lang.de.stop_words.STOP_WORDS)

spacy_stopwords_en = list(spacy.lang.en.stop_words.STOP_WORDS)

#spacy_stopwords_xx = list(spacy.lang.xx.stop_words.STOP_WORDS)
spacy_stopwords_en[0:10]
def spacy_tokenizer(sentence):

    return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)]
# New stop words list 

stop_words = [

    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 

    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'fig', 'fig.', 'al.', 

    'di', 'la', 'il', 'del', 'le', 'della', 'dei', 'delle', 'una', 'da',  'dell',  'non', 'si'

]



customize_stop_words = stop_words + spacy_stopwords_fr + spacy_stopwords_es + spacy_stopwords_it + spacy_stopwords_de + spacy_stopwords_en



# Mark them as stop words

for w in customize_stop_words:

    nlp.vocab[w].is_stop = True
print(customize_stop_words[1:10])

print(len(customize_stop_words))
sent = "Je suis un champion 9 123 *."

spacy_tokenizer(sent)
# vectorizer = CountVectorizer(ngram_range=(1,2), tokenizer = spacy_tokenizer, max_features=800000, analyzer='word')

# data_vectorized = vectorizer.fit_transform(tqdm(all_texts))



# joblib.dump(vectorizer, '/kaggle/working/bigram_vectorizer.csv')

# joblib.dump(data_vectorized, '/kaggle/working/bigram_data_vectorized.csv')
# Load trained vectorizer and vectorized data

vectorizer = joblib.load('/kaggle/input/bigram_vectorizer.csv')

data_vectorized = joblib.load('/kaggle/input/bigram_data_vectorized.csv')
print(data_vectorized.shape)

print(data_vectorized)
# most frequent words

word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'count': np.asarray(data_vectorized.sum(axis=0))[0]})



word_count.sort_values('count', ascending=False).set_index('word')[:20].sort_values('count', ascending=True).plot(kind='barh')

plt.show()
# Define Search Param

# from sklearn.model_selection import GridSearchCV



# search_params = {'n_components': [10, 20, 30, 40, 50], 'learning_decay': [.5, .7, .9]}



# # Init the Model

# lda = LatentDirichletAllocation()



# # Init Grid Search Class

# model = GridSearchCV(lda, param_grid=search_params,n_jobs= -1)



# # Do the Grid Search

# model.fit(data_vectorized)
# Best Model

# best_lda_model = model.best_estimator_



# # Model Parameters

# print("Best Model's Params: ", model.best_params_)



# # Log Likelihood Score

# print("Best Log Likelihood Score: ", model.best_score_)



# # Perplexity

# print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
# lda = LatentDirichletAllocation(n_components=10, learning_decay=0.9,  random_state=0,verbose=1)

# lda.fit(data_vectorized)

# joblib.dump(lda, '/kaggle/working/lda_10.csv')
lda = joblib.load('/kaggle/input/lda_10.csv')
print(lda)
# Visualize topic model 

#pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer)
# Log Likelyhood: Higher the better

print("Log Likelihood: ", lda.score(data_vectorized))



# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)

print("Perplexity: ", lda.perplexity(data_vectorized))



# # See model parameters

# pprint(lda.get_params())
def print_top_words(model, vectorizer, n_top_words):

    feature_names = vectorizer.get_feature_names()

    for topic_idx, topic in enumerate(model.components_):

        message = "\nTopic #%d: " % topic_idx

        message += " ".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        print(message)

    print()
print_top_words(lda, vectorizer, n_top_words=25)
# doc_topic_dist = pd.DataFrame(lda.transform(data_vectorized))

# doc_topic_dist.to_csv('/kaggle/working/doc_topic_dist_10.csv', index=False)
doc_topic_dist = pd.read_csv('/kaggle/input/doc_topic_dist_10.csv')
print(doc_topic_dist.shape)
doc_topic_dist.head()
pca = PCA (n_components=10)

pca_topics = pca.fit(doc_topic_dist).transform(doc_topic_dist)
print ('First component explain {} variance of data and second component explain {} variance of data'.format(pca.explained_variance_ratio_[0],pca.explained_variance_ratio_[2]))
plt.scatter(pca_topics[:,0],pca_topics[:,1])

plt.title("PCA")

plt.show()
df.index = range(len(df.index))

df.tail()
is_covid19_article = df.text.str.contains('COVID-19|SARS-CoV-2|2019-nCov|SARS Coronavirus 2|2019 Novel Coronavirus')

print(is_covid19_article.shape)

print(is_covid19_article)
doc_topic_dist[is_covid19_article]
def get_k_nearest_docs(doc_dist, k=5, lower=1950, upper=2020, only_covid19=False, get_dist=False):

    '''

    doc_dist: topic distribution (sums to 1) of one article

    

    Returns the index of the k nearest articles (as by Jensen–Shannon divergence in topic space). 

    '''

    

    relevant_time = df.publish_year.between(lower, upper)

    

    if only_covid19:

        temp = doc_topic_dist[relevant_time & is_covid19_article]

        

    else:

        temp = doc_topic_dist[relevant_time]

         

    distances = temp.apply(lambda x: jensenshannon(x, doc_dist), axis=1)

    k_nearest = distances[distances != 0].nsmallest(n=k).index

    

    if get_dist:

        k_distances = distances[distances != 0].nsmallest(n=k)

        return k_nearest, k_distances

    else:

        return k_nearest
task5 = ["Effectiveness of drugs being developed and tried to treat COVID-19 patients. Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.",

"Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.",

"Exploration of use of best animal models and their predictive value for a human vaccine.",

"Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.",

"Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.",

"Efforts targeted at a universal coronavirus vaccine.",

"Efforts to develop animal models and standardize challenge studies",

"Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers",

"Approaches to evaluate risk for enhanced disease after vaccination",

"Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]"]



task5b = [

"Exploration of use of best animal models and their predictive value for a human vaccine."]



tasks={'What do we know about vaccines and therapeutics?': task5, 

       'Exploration of use of best animal models and their predictive value for a human vaccine.': task5b}
def relevant_articles(tasks, k=3, lower=1950, upper=2020, only_covid19=False):

    tasks = [tasks] if type(tasks) is str else tasks 

    

    tasks_vectorized = vectorizer.transform(tasks)

    tasks_topic_dist = pd.DataFrame(lda.transform(tasks_vectorized))



    for index, bullet in enumerate(tasks):

        print(bullet)

        recommended = get_k_nearest_docs(tasks_topic_dist.iloc[index], k, lower, upper, only_covid19)

        recommended = df.iloc[recommended].copy()



    return recommended[['paper_id','title','authors','affiliations','abstract','text','source_x','journal','pubmed_id','publish_time']]
relevant_articles(task5b, 10, only_covid19=True)
recommended = relevant_articles(task5b, 10, only_covid19=True)
title = recommended.iloc[3,].title

abstract = recommended.iloc[3,].abstract

text = recommended.iloc[3,].text

print("Title: {0}  \nAbstract: {1}".format(title,abstract))
title = recommended.iloc[4,].title

abstract = recommended.iloc[4,].abstract

text = recommended.iloc[4,].text[0:500]

print("Title: {0}  \nAbstract: {1}".format(title,abstract))
title = recommended.iloc[6,].title

abstract = recommended.iloc[6,].abstract

text = recommended.iloc[6,].text[0:500]

print("Title: {0}  \nAbstract: {1}".format(title,abstract))
title = recommended.iloc[8,].title

abstract = recommended.iloc[8,].abstract

text = recommended.iloc[8,].text[0:500]

print("Title: {0}  \nAbstract: {1}".format(title,abstract))
title = recommended.iloc[3,].title

abstract = recommended.iloc[3,].abstract

text = recommended.iloc[3,].text

print("Title: {0}  \nAbstract: {1}".format(title,abstract))
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)

from collections import defaultdict

from gensim import corpora

from gensim import models

from gensim import similarities

import nltk

from nltk import tokenize
nltk.download('punkt')
documents = tokenize.sent_tokenize(text)

print(documents[0])
documents = tokenize.sent_tokenize(text)



# remove common words and tokenize

stoplist = [

    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 

    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'fig', 'fig.', 'al.', 

    'di', 'la', 'il', 'del', 'le', 'della', 'dei', 'delle', 'una', 'da',  'dell',  'non', 'si'

]

texts = [

    [word for word in document.lower().split() if word not in stoplist]

    for document in documents

]



# remove words that appear only once

frequency = defaultdict(int)

for text in texts:

    for token in text:

        frequency[token] += 1



texts = [

    [token for token in text if frequency[token] > 1]

    for text in texts

]



dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]
print(dictionary)
# Similarity interface

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)



doc = "Exploration of use of best animal models and their predictive value for a human vaccine."

vec_bow = dictionary.doc2bow(doc.lower().split())

vec_lsi = lsi[vec_bow]  # convert the query to LSI space

print(vec_lsi)
index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it



sims = index[vec_lsi]  # perform a similarity query against the corpus

#print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples



sims = sorted(enumerate(sims), key=lambda item: -item[1])

for i, s in enumerate(sims):

    print(s, documents[i])
import logging

logger = logging.getLogger()

logger.setLevel(logging.CRITICAL)

pd.set_option('display.max_colwidth', -1)
#pd.reset_option('^display.', silent=True)
# Refresh corpus

df= pd.read_csv('/kaggle/input/all_papers_filtered.csv')
#Implement function for sentance matching

def sim_sentence(title,text, stopwords, search):

    """"

    title: str title of paper with text

    text: str body of text of paper you want to search

    stopwords: list of stopwords to remove from text

    search: str search of sentence you want to return a match for.

    

    returns dataframe of document number and similarity score to search string

    """

    

    documents = tokenize.sent_tokenize(text)



    # remove common words and tokenize

    stoplist = stopwords 

    

    texts = [

        [word for word in document.lower().split() if word not in stoplist]

        for document in documents

    ]



    # remove words that appear only once

    frequency = defaultdict(int)

    for text in texts:

        for token in text:

            frequency[token] += 1



    texts = [

        [token for token in text if frequency[token] > 1]

        for text in texts

    ]



    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]

    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

    

    doc = search

    vec_bow = dictionary.doc2bow(doc.lower().split())

    vec_lsi = lsi[vec_bow]  # convert the query to LSI space

    #print(vec_lsi)

    

    index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it

    sims = index[vec_lsi]  # perform a similarity query against the corpus

    #print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples



    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    top10_sentences = sims[0:10]

    df=pd.DataFrame()

    for i, s in enumerate(top10_sentences):

        index = top10_sentences[i][0]

        sim_score = top10_sentences[i][1]

        matched_sentence = documents[i]

        d= {"title": title,"Index": index,"sim_score":sim_score,"matched_sentence":matched_sentence}

        df= df.append(d,ignore_index=True)

    return df[['Index','title','matched_sentence','sim_score']]





def relevant_sentences(task, search, stopwords, k=3,lower=1950, upper=2020, only_covid19=False):

    """"

    task: list of task that you want to return relevant sentences for

    search: search str of task 

    stopwords: list of stopwords

    k = number of recommended articles to return

    lower= beginning year range

    upper= end year range

    only_covid19= boolean indicating whether to only return covid-19 articles

    

    Returns: printed dataframes showing matched sentances and similarity scores to task and search string

    """

    # Return recommended articles

    recommended = relevant_articles(task, k, only_covid19)

    stoplist= stopwords

    doc = search

    

    # Run for all papers in the recommended papers

    for index,row in recommended.iterrows():

        print(row['title'])

        df = sim_sentence(row['title'],row['text'],stoplist,doc)

        print(df[['matched_sentence','sim_score']].head(10))
#Test out function

search = "Exploration of use of best animal models and their predictive value for a human vaccine."

relevant_sentences(task5b, search, customize_stop_words,k=10,only_covid19= True)