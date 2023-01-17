import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.tokenize import WhitespaceTokenizer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

import re

import numpy as np

import pandas as pd

from pprint import pprint



# NLTK Stop words

from nltk.corpus import stopwords

import nltk



# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel



from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel



# spacy for lemmatization

import spacy



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



# 1. Wordcloud of Top N words in each topic

from matplotlib import pyplot as plt

from wordcloud import WordCloud, STOPWORDS

import matplotlib.colors as mcolors



from collections import Counter



from matplotlib.ticker import FuncFormatter



# Get topic weights and dominant topics ------------

from sklearn.manifold import TSNE

from bokeh.plotting import figure, output_file, show

from bokeh.models import Label

from bokeh.io import output_notebook

stop_words = stopwords.words('english')

stop_words.extend(['verify','verified','trip'])



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
vect_file = open("../input/svm-model/vectorizer.sav",'rb')

classi_file = open("../input/svm-model/classifier.sav",'rb')

vect = pickle.load(vect_file)

classi = pickle.load(classi_file)
air_india_express = pd.read_excel('../input/airlines/Air-India-Express_f.xlsx')

air_asia_india = pd.read_excel('../input/airlines/AirAsia-India_f.xlsx')

goair = pd.read_excel('../input/airlines/GoAir_f.xlsx')

indigo = pd.read_excel('../input/airlines/IndiGo_f.xlsx')

spicejet = pd.read_excel('../input/airlines/spicejet_data_f.xlsx')

vistara = pd.read_excel('../input/airlines/vistara_f.xlsx')
vistara
def clean_text(text):

    # lower text

    text = text.lower()

    text = " ".join([word for word,tag in nltk.tag.pos_tag(text.split()) if tag not in ['NNP','NNPS']])

    # tokenize text and remove puncutation

    text = [word.strip(string.punctuation) for word in text.split(" ")]

    # remove words that contain numbers

    text = [word for word in text if not any(c.isdigit() for c in word)]

    # remove stop words

    stop = stopwords.words('english')

    stop.extend(['âœ…', 'verify','trip','verified'])

    text = [x for x in text if x not in stop]

    # remove empty tokens

    text = [t for t in text if len(t) > 0]

    # pos tag text

    pos_tags = pos_tag(text)

    # lemmatize text

    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    # remove words with only one letter

    text = [t for t in text if len(t) > 1]

    # join all

    text = " ".join(text)

    return(text)





def get_wordnet_pos(pos_tag):

    if pos_tag.startswith('J'):

        return wordnet.ADJ

    elif pos_tag.startswith('V'):

        return wordnet.VERB

    elif pos_tag.startswith('N'):

        return wordnet.NOUN

    elif pos_tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN
air_india_express.dropna(inplace = True)

air_asia_india.dropna(inplace = True)

goair.dropna(inplace = True)

indigo.dropna(inplace = True)

spicejet.dropna(inplace = True)

vistara.dropna(inplace = True)
air_india_express['clean review'] = air_india_express['review'].apply(lambda x: clean_text(x))

air_asia_india['clean review'] = air_asia_india['review'].apply(lambda x: clean_text(x))

goair['clean review'] = goair['review'].apply(lambda x: clean_text(x))

indigo['clean review'] = indigo['review'].apply(lambda x: clean_text(x))

spicejet['clean review'] = spicejet['review'].apply(lambda x: clean_text(x))

vistara['clean review'] = vistara['review'].apply(lambda x: clean_text(x))
# air india express

vectorized_review = vect.transform(air_india_express['clean review'].tolist())

air_india_express['predicted'] = classi.predict(vectorized_review)



#air asia india

vectorized_review = vect.transform(air_asia_india['clean review'].tolist())

air_asia_india['predicted'] = classi.predict(vectorized_review)



#goair

vectorized_review = vect.transform(goair['clean review'].tolist())

goair['predicted'] = classi.predict(vectorized_review)



#indigo

vectorized_review = vect.transform(indigo['clean review'].tolist())

indigo['predicted'] = classi.predict(vectorized_review)



#spicejet

vectorized_review = vect.transform(spicejet['clean review'].tolist())

spicejet['predicted'] = classi.predict(vectorized_review)



#vistara

vectorized_review = vect.transform(vistara['clean review'].tolist())

vistara['predicted'] = classi.predict(vectorized_review)

spicejet
from nltk import ngrams

def ngrams(input_list):

    #onegrams = input_list

    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]

    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]

    return bigrams+trigrams
air_india_express['grams'] = air_india_express['clean review'].apply(lambda x:ngrams(x.split()))

air_asia_india['grams'] = air_asia_india['clean review'].apply(lambda x:ngrams(x.split()))

goair['grams'] = goair['clean review'].apply(lambda x:ngrams(x.split()))

indigo['grams'] = indigo['clean review'].apply(lambda x:ngrams(x.split()))

spicejet['grams'] = spicejet['clean review'].apply(lambda x:ngrams(x.split()))

vistara['grams'] = vistara['clean review'].apply(lambda x:ngrams(x.split()))
indigo
import collections

def count_words(input):

    cnt = collections.Counter()

    for row in input:

        for word in row:

            cnt[word] += 1

    return cnt
count_words(air_india_express[(air_india_express.predicted == 1)]['grams']).most_common(10)
d = dict(count_words(air_india_express[(air_india_express.predicted == 1)]['grams']))

import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(height= 350, width = 550)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize= (15,15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
count_words(air_india_express[(air_india_express.predicted == 0)]['grams']).most_common(10)
d = dict(count_words(air_india_express[(air_india_express.predicted == 0)]['grams']))

import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(height= 350, width = 550)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize= (15,15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
count_words(air_asia_india[(air_asia_india.predicted == 1)]['grams']).most_common(10)
d = dict(count_words(air_asia_india[(air_asia_india.predicted == 1)]['grams']))

import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(height= 350, width = 550)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize= (15,15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
count_words(air_asia_india[(air_asia_india.predicted == 0)]['grams']).most_common(10)
d = dict(count_words(air_asia_india[(air_asia_india.predicted == 0)]['grams']))

import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(height= 350, width = 550)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize= (15,15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
count_words(indigo[(indigo.predicted == 1)]['grams']).most_common(10)
d = dict(count_words(indigo[(indigo.predicted == 1)]['grams']))

import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(height= 350, width = 550)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize= (15,15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
count_words(indigo[(indigo.predicted == 0)]['grams']).most_common(10)
d = dict(count_words(indigo[(indigo.predicted == 0)]['grams']))

import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(height= 350, width = 550)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize= (15,15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
count_words(spicejet[(spicejet.predicted == 1)]['grams']).most_common(10)
d = dict(count_words(spicejet[(spicejet.predicted == 1)]['grams']))

import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(height= 350, width = 550)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize= (15,15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
count_words(spicejet[(spicejet.predicted == 0)]['grams']).most_common(10)
d = dict(count_words(spicejet[(spicejet.predicted == 0)]['grams']))

import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(height= 350, width = 550)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize= (15,15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
count_words(vistara[(vistara.predicted == 1)]['grams']).most_common(10)
d = dict(count_words(vistara[(vistara.predicted == 1)]['grams']))

import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(height= 350, width = 550)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize= (15,15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
count_words(vistara[(vistara.predicted == 0)]['grams']).most_common(10)
d = dict(count_words(vistara[(vistara.predicted == 0)]['grams']))

import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(height= 350, width = 550)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize= (15,15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
count_words(goair[(goair.predicted == 1)]['grams']).most_common(10)
count_words(goair[(goair.predicted == 0)]['grams']).most_common(10)
d = dict(count_words(goair[(goair.predicted == 0)]['grams']))

import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(height= 350, width = 550)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize= (15,15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()