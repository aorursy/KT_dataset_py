# Importing alll libraries and packages

import pandas as pd
import numpy as np
import scipy as sp
import sklearn
import sys
from nltk.corpus import stopwords
import nltk
from gensim.models import ldamodel
import gensim.corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import string
from nltk.stem.wordnet import WordNetLemmatizer
import re
# from gensim.test.utils import common_corpus, common_dictionary
# from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from gensim import matutils, models
import scipy.sparse
from nltk import word_tokenize, pos_tag
# from sklearn.feature_extraction import text
import pandas as pd
Train=pd.read_csv('data.csv')
pd.set_option('display.max_colwidth',150)
Train.head(6)
# removing userid email and numbers if present
Train['clean_text']=Train['text'].str.lower().apply(lambda x: re.sub(r'(@[\S]+)|(\w+:\/\/\S+)|(\d+)','',x))


# removing stopwords and special character and returned lemmatized word
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop and len(i)>1])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
Train['clean_text']=Train['clean_text'].apply(lambda x: clean(x))
Train.head(6)
# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(Train.clean_text)
data = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names(),index=Train['Id'])
# data_dtm.index = Train['Id'].index
# One of the required inputs is a term-document matrix
tdm = data.transpose()
data.head()
# We're going to put the term-document matrix into a new gensim format, from data --> sparse matrix --> gensim corpus
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
id2word = dict((v, k) for k, v in cv.vocabulary_.items())
id2word
# Fit the model for 5 topics
lda = models.LdaModel(corpus=corpus, num_topics=5, id2word=id2word, passes=100,eta=.90)
# get the list of top 20 words in each topic after applying LDA model
def get_lda_topics(model, num_topics,num_words):
    word_dict = {}
    topics = model.show_topics(num_topics,num_words)
    word_dict = {'Topic '+str(i):[x.split('*') for x in words.split('+')] \
                 for i,words in model.show_topics(num_topics,num_words)}
    return pd.DataFrame.from_dict(word_dict)

get_lda_topics(lda,5,20)
corpus_transformed = lda[corpus]
# getting topic having maximum score for a document
topic=[]
for i in range(len(corpus_transformed)):
    v=dict(corpus_transformed[i])
    for top, score in v.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if score == max(v.values()):
            topic.append(top)
id_topics=pd.DataFrame([a for a in topic],index=data.index)
id_topics.columns=['Topics']

# "glassdoor_reviews"
# "tech_news"
# "room_rentals"
# "sports_news"
# "Automobiles"
id_topics['topic']=np.where(id_topics['Topics']==0,'tech_news',
                                np.where(id_topics['Topics']==1,'glassdoor_reviews',
                                        np.where(id_topics['Topics']==2,'sports_news',
                                                np.where(id_topics['Topics']==3,'Automobiles','room_rentals'))))
# id_topics.head(20)
final=id_topics.reset_index()
final=final[['Id','topic']]
final.to_csv('final_output12.csv',index=False)
final


## score .94197
## Score .9617

# This is the final output csv having ID and Topic column in it
# set Id as an index in data frame
Train=Train.set_index('Id')
Train.head(4)
# Let's create a function to pull out nouns from a string of text
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2].startswith('N') or pos[:2].startswith('J')
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)
# Apply the nouns function to the transcripts to filter only on nouns
data_nouns_adj = pd.DataFrame(Train.clean_text.apply(nouns_adj))
data_nouns_adj.head(4)
# Creating sparse matrix with ttems as columns and ids as index
cvna = CountVectorizer(stop_words=stop, max_features = 5000, max_df=.8)
data_cvna = cvna.fit_transform(data_nouns_adj.clean_text)
data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = data_nouns_adj.index
data_dtmna.head(4)
# Create the gensim corpus
corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))

# Create the vocabulary dictionary
id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())
# Create LDA model with 5 topics, number of passes is given 50 to get fine tuned result
ldana = models.LdaModel(corpus=corpusna, num_topics=5, id2word=id2wordna, passes=100,eta=.90)
get_lda_topics(ldana,5,20)
corpus_transformed = ldana[corpusna]
# getting topic having maximum score for a document
topic=[]
for i in range(len(corpus_transformed)):
    v=dict(corpus_transformed[i])
    for top, score in v.items():  
        if score == max(v.values()):
            topic.append(top)
# Get the topic the each document contains

id_topics=pd.DataFrame([a for a in topic],index=data_dtmna.index)
id_topics.columns=['Topics']
# id_topics.head(6)

# "glassdoor_reviews"
# "tech_news"
# "room_rentals"
# "sports_news"
# "Automobiles"
id_topics['topic']=np.where(id_topics['Topics']==0,'sports_news',
                                np.where(id_topics['Topics']==1,'glassdoor_reviews',
                                        np.where(id_topics['Topics']==2,'Automobiles',
                                                np.where(id_topics['Topics']==3,'room_rentals','tech_news'))))
id_topics.head(20)
final=id_topics.reset_index()
final=final[['Id','topic']]
final.to_csv('final5.csv',index=False)
final

# 0.91267
# 0.89135
# Final output having Noun and Adjective in Document
# After comparing both results in this notebook, publish the one having highest score 
# Perplexity, lower the better.
# print('\nPerplexity: ', lda.log_perplexity(corpusna))  
# # Coherance score, higher is better
# coherence_model_lda = CoherenceModel(model=lda, texts=Train['clean_text'], dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)