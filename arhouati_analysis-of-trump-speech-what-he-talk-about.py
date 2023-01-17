# import all needed libraries

import numpy as np

import pandas as pd

import spacy

from spacy import displacy

import matplotlib.pyplot as plt

import warnings

import os



import nltk

from nltk.corpus import words as english_words, stopwords

from nltk.stem import PorterStemmer



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split



import seaborn as sns



from wordcloud import WordCloud, STOPWORDS 



from collections import Counter



import torch



import re



warnings.filterwarnings('ignore')



%matplotlib inline
# load data

speechs = list()

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_path = os.path.join(dirname, filename)

        print(file_path)

        text = open(file_path,'r').read()

        speechs.append(text)



print(f"Total Number of Documents : {len(speechs)}")
def cleansing_text(text: str) -> str:

    ## replacing the newlines and extra spaces, and change all character to lower case

    corpus = text.replace('\n', ' ').replace('\r', '').replace('  ',' ').lower()



    ## removing everything except alphabets

    corpus_sans_symbols = re.sub('[^a-zA-Z \n]', '', corpus)



    ## removing stopwords

    stop_words = set(w.lower() for w in stopwords.words())



    corpus_sans_symbols_stopwords = ' '.join(filter(lambda x: x.lower() not in stop_words, corpus_sans_symbols.split()))

    return corpus_sans_symbols_stopwords

    

preprocessed_speechs = list(map(cleansing_text, speechs))



# display an example of Tramp Speech

print(preprocessed_speechs[0][:100] + '...')

stemmer = nltk.PorterStemmer()

def stemmer_str(text: str)-> str:

    corpus_stemmed = ' ' .join (map(lambda str: stemmer.stem(str), text.split()))

    return corpus_stemmed



preprocessed_speechs_stemmer = list(map(stemmer_str, preprocessed_speechs))



print(preprocessed_speechs_stemmer[0][:100] + '...')

all_speechs_str = ' '.join(preprocessed_speechs)



wordcloud = WordCloud(width = 800, height = 800,background_color ='grey', min_font_size = 10).generate(all_speechs_str)

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.rcParams.update({'font.size': 25})

plt.axis("off") 

plt.title('Word Cloud: DJT Rallies ')

plt.tight_layout(pad = 0) 

  

plt.show()
word_freq_count = Counter(' '.join(preprocessed_speechs).split(" "))



common_words = [word[0] for word in word_freq_count.most_common(20)]

common_counts = [word[1] for word in word_freq_count.most_common(20)]



plt.figure(figsize=(15, 12))



sns.set_style("whitegrid")

sns_bar = sns.barplot(x=common_words, y=common_counts)

sns_bar.set_xticklabels(common_words, rotation=45)

plt.title('Most Common Words in the document')

plt.show()
all_speechs_str = ' '.join(preprocessed_speechs_stemmer)



wordcloud = WordCloud(width = 800, height = 800,background_color ='grey', min_font_size = 10).generate(all_speechs_str)

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.rcParams.update({'font.size': 25})

plt.axis("off") 

plt.title('Word Cloud: DJT Rallies ')

plt.tight_layout(pad = 0) 

  

plt.show()

 
word_freq_count = Counter(' '.join(preprocessed_speechs_stemmer).split(" "))



common_words = [word[0] for word in word_freq_count.most_common(20)]

common_counts = [word[1] for word in word_freq_count.most_common(20)]



plt.figure(figsize=(15, 12))



sns.set_style("whitegrid")

sns_bar = sns.barplot(x=common_words, y=common_counts)

sns_bar.set_xticklabels(common_words, rotation=45)

plt.title('Most Common Words in the document')

plt.show()
from itertools import islice



tfidf_vec = TfidfVectorizer(stop_words="english")

transformed = tfidf_vec.fit_transform(raw_documents=preprocessed_speechs)

index_value={i[1]:i[0] for i in tfidf_vec.vocabulary_.items()}

print( {k: index_value[k] for k in list(index_value)[:50]})
tfidf_vec = TfidfVectorizer(stop_words="english")

transformed = tfidf_vec.fit_transform(raw_documents=preprocessed_speechs_stemmer)

index_value={i[1]:i[0] for i in tfidf_vec.vocabulary_.items()}

print( {k: index_value[k] for k in list(index_value)[:50]})
# this function can get any n-grams from a string

# note that we can use also nltk.bigrams(eng_tokens)

def get_n_grams(text: str, n:int):

    n_grams = list()

    text_tokens = text.split(' ')

    for index, token in enumerate(text_tokens):

        if index+n < len(text_tokens):

            n_grams.append(tuple(text_tokens[index:index+n]))

    return n_grams
#get bi-grams from all speechs

bi_grams = list()

for speech in preprocessed_speechs:

    bi_grams = bi_grams + get_n_grams(speech, 2)



bi_grams_freq = nltk.FreqDist(bi_grams)

bi_grams_sorted = sorted(bi_grams_freq , key = bi_grams_freq.__getitem__, reverse = True)



# keep only 20

bi_grams_sorted = bi_grams_sorted[:20]

[print(item, ' : ', bi_grams_freq[item]) for item in bi_grams_sorted]



bi_grams_dict = dict()

for item in bi_grams_sorted:

    bi_grams_dict[' '.join(item)] = bi_grams_freq[item]

 

wordcloud = WordCloud(width = 800, height = 800,background_color ='grey', min_font_size = 10).generate_from_frequencies(bi_grams_dict)

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.rcParams.update({'font.size': 25})

plt.axis("off") 

plt.title('Word Cloud: DJT Rallies ')

plt.tight_layout(pad = 0) 

  

plt.show()
#get 3-grams from all speechs

three_grams = list()

for speech in preprocessed_speechs:

    three_grams = three_grams + get_n_grams(speech, 3)



three_grams_freq = nltk.FreqDist(three_grams)

three_grams_sorted = sorted(three_grams_freq , key = three_grams_freq.__getitem__, reverse = True)



# keep only 20

three_grams_sorted = three_grams_sorted[:20]

[print(item, ' : ', three_grams_freq[item]) for item in three_grams_sorted]



three_grams_dict = dict()

for item in three_grams_sorted:

    three_grams_dict[' '.join(item)] = three_grams_freq[item]

     

wordcloud = WordCloud(width = 800, height = 800,background_color ='grey', min_font_size = 10).generate_from_frequencies(three_grams_dict)

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.rcParams.update({'font.size': 25})

plt.axis("off") 

plt.title('Word Cloud: DJT Rallies ')

plt.tight_layout(pad = 0) 

  

plt.show()