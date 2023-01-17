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
## Importing Libraries

import numpy as np

import spacy

from spacy import displacy

import matplotlib.pyplot as plt

import warnings

import os

warnings.filterwarnings('ignore')

%matplotlib inline
dictOfFilenames={i : filenames[i] for i in range(0, len(filenames) )}

dict_files=dictOfFilenames.copy()

dict_files
for i,filename in enumerate(filenames):

    dictOfFilenames[i] = open(os.path.join(dirname, filename),'r').read()
dictOfFilenames[0]
import nltk

from nltk.corpus import words as english_words, stopwords

import re



## replacing the newlines and extra spaces

corpus = dictOfFilenames[0].replace('\n', ' ').replace('\r', '').replace('  ',' ').lower()



## removing everything except alphabets

corpus_sans_symbols = re.sub('[^a-zA-Z \n]', '', corpus)



## removing stopwords

stop_words = set(w.lower() for w in stopwords.words())



corpus_sans_symbols_stopwords = ' '.join(filter(lambda x: x.lower() not in stop_words, corpus_sans_symbols.split()))

print (corpus_sans_symbols_stopwords)
from nltk.stem import PorterStemmer

stemmer=nltk.PorterStemmer()

corpus_stemmed = ' ' .join (map(lambda str: stemmer.stem(str), corpus_sans_symbols_stopwords.split()))

print (corpus_stemmed)
# Plot top 20 frequent words

from collections import Counter

word_freq = Counter(corpus_stemmed.split(" "))

import seaborn as sns

sns.set_style("whitegrid")

common_words = [word[0] for word in word_freq.most_common(20)]

common_counts = [word[1] for word in word_freq.most_common(20)]





plt.figure(figsize=(12, 8))



sns_bar = sns.barplot(x=common_words, y=common_counts)

sns_bar.set_xticklabels(common_words, rotation=45)

plt.title('Most Common Words in the document')

plt.show()
import spacy

## Spacy example 

nlp = spacy.load('en')

doc = nlp(dictOfFilenames[0])

token_list=[]

for token in doc:

    token_list.append([token.text,token.idx,token.lemma_,token.is_punct,token.pos_,token.tag_])

print(token_list[:20])
## passing our text into spacy

doc = nlp(dictOfFilenames[0])



## filtering stopwords, punctuations, checking for alphabets and capturing the lemmatized text

spacy_tokens = [token.lemma_ for token in doc if token.is_stop != True \

                and token.is_punct != True and token.is_alpha ==True]
word_freq_spacy = Counter(spacy_tokens)



# Plot top 20 frequent words



sns.set_style("whitegrid")

common_words = [word[0] for word in word_freq_spacy.most_common(20)]

common_counts = [word[1] for word in word_freq_spacy.most_common(20)]





plt.figure(figsize=(12, 8))



sns_bar = sns.barplot(x=common_words, y=common_counts)

sns_bar.set_xticklabels(common_words, rotation=45)

plt.title('Most Common Words in the document')

plt.show()
text_str = ''.join(dictOfFilenames[0].replace('\n',' ').replace('\t',' '))

sentences_split = text_str.split(".")

sentences_split[67]
doc = nlp(text_str)

sentence_list = [s for s in doc.sents]

sentence_list[67]
spacy.displacy.render(sentence_list[67], style='dep',jupyter=True,options = {'compact':60})

pos_list = [(token, token.pos_) for token in sentence_list[67]]
text_ent_example=dictOfFilenames[0]
doc = nlp(text_ent_example)

spacy.displacy.render(doc, style='ent',jupyter=True)
# Obtain additional stopwords from nltk

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
stop_words
import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS
# Remove stopwords and remove words with 2 or less characters

def preprocess(text):

    result = []

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:

            result.append(token)

            

    return result
type(dictOfFilenames[0])
df = pd.DataFrame([dictOfFilenames[0]])
df['clean'] = df[0].apply(preprocess)
df.head()
# Obtain the total words present in the dataset

list_of_words = []

for i in df.clean:

    for j in i:

        list_of_words.append(j)
list_of_words
len(list_of_words)
# Obtain the total number of unique words

total_words = len(list(set(list_of_words)))

total_words
# join the words into a string

df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))
df
from wordcloud import WordCloud, STOPWORDS



# plot the word cloud for text for Toledo Rally

plt.figure(figsize = (20,20)) 

wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df.clean_joined))

plt.imshow(wc, interpolation = 'bilinear')