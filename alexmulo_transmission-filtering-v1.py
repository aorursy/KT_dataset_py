import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import nltk

import re

from nltk.corpus import stopwords

from itertools import chain



from nltk import word_tokenize

from scipy import stats
#Import Metadata dataset

path="/kaggle/input/CORD-19-research-challenge/"

all_sources=pd.read_csv(path+"metadata.csv")
all_sources.head(3)
all_sources.isna().sum()
# Set text to lower case

all_sources['title'] = all_sources['title'].str.lower()

all_sources['abstract'] = all_sources['abstract'].str.lower()
# Remove not useful characters 

all_sources['abstract'] = all_sources['abstract'].str.replace(r'[,\!?&$_\{\}\(\)]','')

all_sources['title'] = all_sources['title'].str.replace(r'[,\!?&$_\{\}\(\)]','')
# Drop papers without abstract

all_sources = all_sources[all_sources['abstract'].notna()]
# Filter papers with the word transmission in the abstract

transmission = all_sources[all_sources['abstract'].str.contains(r'transmission')]
stop_words=stopwords.words("english")
# Fuction to make tokens with sentences

def sentence_list(df,col='abstract'):

    transmission_tokens = df[col].apply(lambda x: nltk.tokenize.sent_tokenize(x))

    transmission_tokens = transmission_tokens.values.tolist()

    return list(chain(*transmission_tokens))
# Tokens of senteces 

transmission_sentence = sentence_list(transmission)
def print_sentences(sentences,content):

    sents = [sent for sent in sentences if content in sent]

    print(len(sents))

    print('- ', '\n- '.join(sents))
# Sentences containing the word transmission. However, these are too much sentences to deal with. Therefore, I will try to filter using the specific topics.

print_sentences(transmission_sentence,'transmission')
# Tokens of sentences contaning the word transmission

transmission_sentence2 = [sent for sent in transmission_sentence if 'transmission' in sent]
# filter sentences with related topics

transmission_sentence2_asymptomatic = [sent for sent in transmission_sentence2 if 'asymptomatic' in sent]

transmission_sentence2_Seasonality = [sent for sent in transmission_sentence2 if 'seasonality' in sent]

transmission_sentence2_models = [sent for sent in transmission_sentence2 if 'models' in sent]

transmission_sentence2_secondary = [sent for sent in transmission_sentence2 if 'secondary transmission' in sent]

transmission_sentence2_ppe = [sent for sent in transmission_sentence2 if 'ppe' in sent]

transmission_sentence2_environment = [sent for sent in transmission_sentence2 if 'environment' in sent]
print('Topic Asymptomatic: ',len(transmission_sentence2_asymptomatic))

print('Topic Seasonality: ',len(transmission_sentence2_Seasonality))

print('Topic Models: ',len(transmission_sentence2_models))

print('Topic Secondary Transmission: ',len(transmission_sentence2_secondary))

print('Topic PPE: ',len(transmission_sentence2_ppe))

print('Topic Environment: ',len(transmission_sentence2_environment))
for se in transmission_sentence2_asymptomatic:

    print('- ',se,'\n')
for se in transmission_sentence2_Seasonality:

    print('- ',se,'\n')
for se in transmission_sentence2_models:

    print('- ',se,'\n')
for se in transmission_sentence2_secondary:

    print('- ',se,'\n')
for se in transmission_sentence2_ppe:

    print('- ',se,'\n')
for se in transmission_sentence2_environment:

    print('- ',se,'\n')