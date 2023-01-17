import re

import numpy as np

import pandas as pd

from pprint import pprint



# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel



# Plotting tools

import pyLDAvis

import pyLDAvis.gensim  # don't skip this

import matplotlib.pyplot as plt

%matplotlib inline
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
import nltk

from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize , RegexpTokenizer

from nltk.corpus import stopwords



from nltk.stem import PorterStemmer

stemmer = PorterStemmer()



from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()



from nltk.corpus import wordnet



from nltk.util import ngrams

from collections import Counter

import scipy as sp;

import sklearn;

import sys;

from nltk.corpus import stopwords;

from gensim.models import ldamodel

import gensim.corpora;

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;

from sklearn.decomposition import NMF;

from sklearn.preprocessing import normalize;

import pickle;
"glassdoor_reviews"



"tech_news"



"room_rentals"



"sports_news"



"Automobiles"
df = pd.read_csv('/kaggle/input/unstructured-l0-nlp-hackathon/data.csv')

df.head()
sents = df['text']
sents[4]
# Answer 1: Text cleaning



import string

string.punctuation



# NLTK Stop words

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

stop_words.extend(['from', 'subject', 're', 'edu', 'use'])



# print(stop_words)



def cleantext(sent):

    words = sent.split()

    dum1 = ' '.join([a for a in words if a not in stop_words])

    out = re.sub('[0-9]+','',dum1)

    return out



def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations





data = list(map(cleantext,sents))

data_words = list(sent_to_words(data))
print(data_words[2])
cleantext(sents[2])
cleantext(sents[0])
df['clean_text'] = data_words
df.head()
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# Define lemmatization



def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    """https://spacy.io/api/annotation"""

    texts_out = []

    for sent in texts:

        doc = nlp(" ".join(sent)) 

        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out
# Do lemmatization keeping only noun, adj, vb, adv

data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])



print(data_lemmatized[:1])
# Create Dictionary

id2word = corpora.Dictionary(data_lemmatized)



# Create Corpus

texts = data_lemmatized



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]



# View

print(corpus[:1])
# Human readable format of corpus (term-frequency)

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
# Build LDA model

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=5, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=100,

                                           passes=10,

                                           alpha=0.2,

                                            eta=0.3,

                                           per_word_topics=True)
# Print the Keyword in the 10 topics

pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]
"glassdoor_reviews"



"tech_news"



"room_rentals"



"sports_news"



"Automobiles"
# Compute Perplexity

print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.



# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
from sklearn.model_selection import GridSearchCV 

from gensim.sklearn_api import LdaTransformer

num_topics = 5

# Define Search Param 

search_params = {'alpha':[0,0.3,0.6,0.9] , 'eta':  [0,0.3,0.6,0.9]   } 

# Init the Model 

lda = LdaTransformer(num_topics=num_topics,id2word=id2word,  random_state=0) 

# Init Grid Search Class 

model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search 

model.fit(corpus)
model.best_estimator_
model.best_params_
lda_model.get_document_topics(corpus[1])
# Build LDA model

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=5, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=2000,

                                           passes=10,

                                           alpha=0.3,

                                            eta=0.6,

                                           per_word_topics=True,

                                           gamma_threshold=0.001)
# Print the Keyword in the 10 topics

pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]
pd.DataFrame(lda_model.get_document_topics(corpus[1]))[1].idxmax(axis=0)
len(corpus)
l1=[]

l2=[]

for i in range(0,len(corpus)):

    tag = pd.DataFrame(lda_model.get_document_topics(corpus[i]))[1].idxmax(axis=0)

    tag1 = ''

    if tag==0:

        tag1 = "glassdoor_reviews"

    elif tag==1:

        tag1 = "room_rentals"

    elif tag==2:

        tag1 = "sports_news"

    elif tag==3:

        tag1 = "tech_news"

    elif tag==4:

        tag1 = "Automobiles"

    l1.append(tag)

    l2.append(tag1)

df['tag_num'] = l1

df['topic'] = l2
df
for i in range(0,len(df['text'])):

    if re.search('bdrm',df['text'][i].lower())!=None:

        print(df['text'][i])

        print(df['topic'][2])
# First submission

df[['Id','topic']].to_csv("Submission_1_0543.csv")
"glassdoor_reviews"



"tech_news"



"room_rentals"



"sports_news"



"Automobiles"
# Compute Perplexity

print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.



# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
import spacy
# Second Iteration:
data_improve = [[re.sub('bdrm','bedroom',y) if y=='bdrm' else y for y in x] for x in data_lemmatized]
data_improve[:20]
# Create Dictionary

id2word = corpora.Dictionary(data_improve)



# Create Corpus

texts = data_improve



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]



# View

print(corpus[:1])
num_topics = 5

# Define Search Param 

search_params = {'alpha':[0,0.2,0.4,0.6,0.8] , 'eta':  [0,0.2,0.4,0.6,0.8]   } 

# Init the Model 

lda = LdaTransformer(num_topics=num_topics,id2word=id2word,  random_state=0) 

# Init Grid Search Class 

model = GridSearchCV(lda, param_grid=search_params)

model.fit(corpus)
model.best_estimator_
# Build LDA model

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=5, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=2000,

                                           passes=10,

                                           alpha=0.2,

                                            eta=0.6,

                                           per_word_topics=True,

                                           gamma_threshold=0.001)
# Print the Keyword in the 10 topics

pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]
l1=[]

l2=[]

for i in range(0,len(corpus)):

    tag = pd.DataFrame(lda_model.get_document_topics(corpus[i]))[1].idxmax(axis=0)

    tag1 = ''

    if tag==0:

        tag1 = "glassdoor_reviews"

    elif tag==1:

        tag1 = "room_rentals"

    elif tag==2:

        tag1 = "tech_news"

    elif tag==3:

        tag1 = "sports_news"

    elif tag==4:

        tag1 = "Automobiles"

    l1.append(tag)

    l2.append(tag1)





df['tag_num1'] = l1

df['topic1'] = l2
"glassdoor_reviews"



"tech_news"



"room_rentals"



"sports_news"



"Automobiles"
# Compute Perplexity

print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.



# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_improve, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
# Third Improvement:

# Guided LDA:

import guidedlda
df.head()

df1 = df[['Id','text']]
import enchant

from enchant.checker import SpellChecker

def spell_check(text):       

    '''

    spell_check: function for correcting the spelling of the reflections

    Expects:  a string

    Returns: a list

    '''

    Corr_RF = []

    #Grab each individual reflection

    for refl in text.split():

        #Check to see if the words are in the dictionary

        chkr = SpellChecker("en_US", refl)

        for err in chkr:

            #for the identified errors or words not in dictionary get the suggested correction

            #and replace it in the reflection string

            if len(err.suggest()) > 0:

                sug = err.suggest()[0]

                err.replace(sug)

        Corr_RF.append(chkr.get_text())

        #return the dataframe with the new corrected reflection column

    return ' '.join(Corr_RF)

data = df1

data['Corrected_content'] = data.text.apply(spell_check)

document = data #to change the name of the dataframe to documents
print(df.text[:5])

print('\n')

print(document.Corrected_content[:5])
from langdetect import detect

def lang_detect(text):

    '''

    lang_detect: function for detecting the language of the reflections

    Expects: a string

    Returns: a list of the detected languages

    '''

    lang = []

    for refl in text:

        lang.append(detect(refl))

    return lang
import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

from nltk.corpus import wordnet

import numpy as np

np.random.seed(42)
document
def lemmatize_stemming(text):

    stemmer = SnowballStemmer('english')

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):

    result = []

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:

            result.append(lemmatize_stemming(token))

    return result

processed_docs = document['Corrected_content'].map(preprocess)
dictionary = gensim.corpora.Dictionary(processed_docs)

count = 0

for k, v in dictionary.iteritems():

    print(k, v)

    count += 1

    if count > 10:

        break

        

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary)
for idx, topic in lda_model.print_topics(-1):

    print('Topic: {} \nWords: {}'.format(idx, topic))
def get_wordnet_pos(word):

    '''tags parts of speech to tokens

    Expects a string and outputs the string and 

    its part of speech'''

    

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)



def word_lemmatizer(text):

    '''lemamtizes the tokens based on their part of speech'''

    

    lemmatizer = WordNetLemmatizer()

    text = lemmatizer.lemmatize(text, get_wordnet_pos(text))

    return text



def reflection_tokenizer(text):

    '''expects a string an returns a list of lemmatized tokens 

        and removes the stop words. Tokens are lower cased and 

        non- alphanumeric characters as well as numbers removed. '''

    text=re.sub(r'[\W_]+', ' ', text) #keeps alphanumeric characters

    text=re.sub(r'\d+', '', text) #removes numbers

    text = text.lower()

    tokens = [word for word in word_tokenize(text)]

    tokens = [word for word in tokens if len(word) >= 3]

    #removes smaller than 3 character

    tokens = [word_lemmatizer(w) for w in tokens]

    tokens = [s for s in tokens if s not in stop_words]

    return tokens
# document

document['lemmatize_token'] = document.Corrected_content.apply(reflection_tokenizer)
from sklearn.feature_extraction.text import CountVectorizer



token_vectorizer = CountVectorizer(tokenizer = reflection_tokenizer, min_df=10, stop_words=stop_words, ngram_range=(1, 4))

X = token_vectorizer.fit_transform(document.Corrected_content)
import guidedlda

tf_feature_names = token_vectorizer.get_feature_names()

word2id = dict((v, idx) for idx, v in enumerate(tf_feature_names))
seed_topic_list= [['pro', 'con', 'work', 'good', 'company', 'people', 'employee', 'management'],

['bedroom', 'bed', 'kitchen', 'home', 'locate', 'location','hotel','bathroom','apartment','house','guesthouse',

'resort'],

['league', 'win', 'team', 'year','player','club'],

['app', 'apple', 'facebook', 'designer'],

['car', 'sedan', 'fast', 'automobile']]
seed_topics = {}

for t_id, st in enumerate(seed_topic_list):

    for word in st:

        seed_topics[word2id[word]] = t_id
model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=10)



model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)
doc_topic = model.transform(X)
len(doc_topic)

# i
for i in range(0,len(doc_topic)):

    print("top topic: {}".format(doc_topic[i].argmax()))
n_top_words = 15

topic_word = model.topic_word_

for i, topic_dist in enumerate(topic_word):

    topic_words = np.array(tf_feature_names)[np.argsort(topic_dist)][:-(n_top_words+1):-1]

    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
l1=[]

l2=[]

for i in range(0,len(doc_topic)):

    # print("top topic: {}".format(doc_topic[i].argmax()))

    tag = doc_topic[i].argmax()

    tag1 = ''

    if tag==0:

        tag1 = "glassdoor_reviews"

    elif tag==1:

        tag1 = "room_rentals"

    elif tag==2:

        tag1 = "sports_news"

    elif tag==3:

        tag1 = "tech_news"

    elif tag==4:

        tag1 = "Automobiles"

    l1.append(tag)

    l2.append(tag1)





document['tag_num1'] = l1

document['topic1'] = l2
document
document[['Id','topic1']].to_csv('Submission_4_0543.csv')
# Create Dictionary

id2word = corpora.Dictionary(data_lemmatized)



# Create Corpus

texts = data_lemmatized



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]



# View

print(corpus[:1])
# Build LDA model

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=10, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=100,

                                           passes=10,

                                           alpha=0.3,

                                            eta=0.6,

                                           per_word_topics=True)
# Print the Keyword in the 10 topics

pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]
"glassdoor_reviews"



"tech_news"



"room_rentals"



"sports_news"



"Automobiles"
# Compute Perplexity

print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.



# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
for i in range(0,len(df['text'])):

    if re.search('traveler',df['text'][i].lower())!=None:

        print(df['text'][i])

        print('\n')
seed_topic_list= [['pro', 'con', 'work', 'good', 'company', 'people', 'employee', 'management'],

['bedroom', 'bed', 'kitchen', 'home', 'locate', 'location','hotel','bathroom','apartment','house','guesthouse',

'resort','traveler','downtown'],

['league', 'win', 'team', 'year','player','club'],

['app', 'apple', 'facebook', 'designer'],

['car', 'sedan', 'fast', 'automobile']]
seed_topics = {}

for t_id, st in enumerate(seed_topic_list):

    for word in st:

        seed_topics[word2id[word]] = t_id
model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=10)



model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)
doc_topic = model.transform(X)
len(doc_topic)

# i
n_top_words = 15

topic_word = model.topic_word_

for i, topic_dist in enumerate(topic_word):

    topic_words = np.array(tf_feature_names)[np.argsort(topic_dist)][:-(n_top_words+1):-1]

    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
l1=[]

l2=[]

for i in range(0,len(doc_topic)):

    # print("top topic: {}".format(doc_topic[i].argmax()))

    tag = doc_topic[i].argmax()

    tag1 = ''

    if tag==0:

        tag1 = "glassdoor_reviews"

    elif tag==1:

        tag1 = "room_rentals"

    elif tag==2:

        tag1 = "sports_news"

    elif tag==3:

        tag1 = "tech_news"

    elif tag==4:

        tag1 = "Automobiles"

    l1.append(tag)

    l2.append(tag1)





document['tag_num3'] = l1

document['topic3'] = l2
document
document[['Id','topic2']].to_csv('Submission_5_0543.csv')
document[['Id','topic3']].to_csv('Submission_6_0543.csv')