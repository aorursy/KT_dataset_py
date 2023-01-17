# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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
df = pd.read_excel('../input/skytrax-airline-reviews/capstone_airline_reviews3.xlsx')
df = df[['customer_review','overall']].copy()
df
df.dropna(inplace = True)

df.reset_index(drop=['index'], inplace = True)
df['label'] = pd.to_numeric(df['overall']).apply(lambda x: 1 if x>6 else 0)
df
def clean_text(text):

    # lower text

    text = text.lower()

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
df['clean review'] = df['customer_review'].apply(lambda x: clean_text(x))

df.label.value_counts().plot.bar()
clean_text("Here is text about an airline I like.")
X_train, X_test, y_train, y_test = train_test_split(df['clean review'], df['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(min_df = 5,

                             max_df = 0.8,

                             sublinear_tf = True,

                             use_idf = True)
train_vectors = vectorizer.fit_transform(X_train)

test_vectors = vectorizer.transform(X_test)
import time

from sklearn import svm

from sklearn.metrics import classification_report

# Perform classification with SVM, kernel=linear

classifier_linear = svm.SVC(kernel='linear')

t0 = time.time()

classifier_linear.fit(train_vectors, y_train)

t1 = time.time()

prediction_linear = classifier_linear.predict(test_vectors)

t2 = time.time()

time_linear_train = t1-t0

time_linear_predict = t2-t1

# results

print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))

report = classification_report(y_test, prediction_linear, output_dict=True)
print('positive: ', report['1'])

print('negative: ', report['0'])
review = ["I received defective piece display is not working properly","It's not even 5 days since i purchased this product.I would say this a specially blended worst Phone in all formats","SUPERB, I AM IN LOVE IN THIS PHONE"]

review_vector = vectorizer.transform(review) # vectorizing

print(classifier_linear.predict(review_vector))
df = pd.read_excel("../input/airplane-review/classification_f.xlsx")
df['clean review'] = df['review'].apply(lambda x:clean_text(x))
df.drop(columns = ['Unnamed: 0','is_bad_review'], inplace = True)
df
review = df['clean review'].tolist()

review_vector = vectorizer.transform(review) # vectorizing

df['Predicted'] = classifier_linear.predict(review_vector)
df.head(20)
from nltk import ngrams

def ngrams(input_list):

    #onegrams = input_list

    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]

    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]

    return bigrams+trigrams

df['grams'] = df['clean review'].apply(lambda x:ngrams(x.split()))

df[['grams']].head()
import collections

def count_words(input):

    cnt = collections.Counter()

    for row in input:

        for word in row:

            cnt[word] += 1

    return cnt
count_words(df[(df.Predicted == 1)]['grams']).most_common(10)
d = dict(count_words(df[(df.Predicted == 1)]['grams']))

import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(height= 350, width = 550)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize= (15,15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()

count_words(df[(df.Predicted == 0)]['grams']).most_common(10)
d = dict(count_words(df[(df.Predicted == 0)]['grams']))

import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(height= 350, width = 550)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize= (15,15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
import pickle

# pickling the vectorizer

pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))

# pickling the model

pickle.dump(classifier_linear, open('classifier.sav', 'wb'))
vect_file = open("../working/vectorizer.sav",'rb')

classi_file = open("../working/classifier.sav",'rb')

vect = pickle.load(vect_file)

classi = pickle.load(classi_file)
vectorized_text = vect.transform(['this was bad airline'])

print(classi.predict(vectorized_text))