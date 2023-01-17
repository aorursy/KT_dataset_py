# Installing 
!pip install praw
!pip install --upgrade gensim
!pip install pyLDAvis
!pip install spacy_langdetect
import pandas as pd
import json
import numpy as np
import datetime 
import re
import string

# NLP Libraries
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy_langdetect import LanguageDetector
import gensim
from gensim import models
from gensim.models import CoherenceModel

# Reddit API
import praw
import praw.models

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt
data = pd.read_csv("/kaggle/input/redditcommentsnetherlandscoronovirus/reddit_nl_coronavirus.csv")
data.head()
print("Number of comments: {}".format(len(data.Comment)))
# Removing Dutch comments
nlp = spacy.load("en")
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)

dutch_comment_index = []
for i,text in enumerate(data.Comment):
    text=nlp(text)
    if(text._.language["language"]=="nl"):
        dutch_comment_index.append(i)

data = data.drop(dutch_comment_index,axis=0)
print("Number of comments after removing dutch comments: {}".format(len(data.Comment)))
# Cleaning comments

# Removing links
def clean_link(text):
    pattern=re.compile(r'http\S+')
    text = re.sub(pattern,"",text)
    return " ".join(text.split())

# Removing numbers and other symbols
def clean_digts_symbols(text):
    pattern = re.compile(r"[^A-Za-z]")
    text = re.sub(pattern," ",text)
    return " ".join(text.split())

# Removing symbols which are not ASCII
def clean_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

# Removing words with less than two letters
def clean_single_words(text):
    return ' '.join(i for i in text.split() if len(i) > 2)

# Transforming text into lowercase
def lower_case(text):
    return text.lower()


data["Comment"] = data.Comment.apply(clean_link).apply(clean_digts_symbols).apply(clean_ascii).apply(clean_single_words).apply(lower_case)
data.head()
# Load English tokenizer, tagger, parser, NER and word vectors
nlp=spacy.load('en')

# List of StopWords, with some added words from my part
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Adding some words to stop_words
stop_words.update(["delete","people","like","remove","know","etc","think","able","hey","don","lol","right","no","yes","thank","talk","thing","look","go","gonna","lot"])

# List with punctuations 
punctuations = string.punctuation

# Function to tokenize the tweets
def spacy_tokenizer(text):
  
  # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(text)

  # Lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.strip() for word in mytokens if word.pos_ in ["PROPN","NOUN"]  ]

  # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

  # Removing words with one or two letters or "..." or "deleted"
    mytokens = [ word for word in mytokens if (len(word)>2)]

  # return preprocessed list of tokens
    return mytokens

data.Comment = data.Comment.apply(spacy_tokenizer)
data.head(15)
dictionary = gensim.corpora.Dictionary(data.Comment)
bow_corpus = [dictionary.doc2bow(text) for text in data.Comment]

lda_model = models.LdaMulticore(bow_corpus, num_topics=25, id2word=dictionary, passes=100, workers=2)
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data.Comment, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
pyLDAvis.enable_notebook()
panel = pyLDAvis.gensim.prepare(lda_model,bow_corpus,dictionary, mds='tsne')
panel