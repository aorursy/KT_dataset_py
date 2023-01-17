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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
amazon_reviews = pd.read_csv('../input/amazonreviews/amazon_reviews.csv')

amazon_reviews.head()
amazon_reviews.shape
reviews = pd.DataFrame()

reviews['reviewText'] = amazon_reviews['reviewText']

reviews['overall'] = amazon_reviews['overall']

reviews.head()
reviews.isnull().sum()
sns.heatmap(reviews.isnull())
reviews = reviews[reviews['reviewText'].notnull()]

reviews.head()
reviews.shape
combined_reviews = " ".join(reviews['reviewText'])

print(combined_reviews[:500])
type(combined_reviews)
from wordcloud import WordCloud

word_cloud = WordCloud(width = 800, height = 800, background_color = 'white', max_words = 150).generate(combined_reviews)

plt.figure(figsize = (12,6))

plt.imshow(word_cloud)

plt.axis('off')

plt.tight_layout(pad = 0)

plt.show()
list1 = []

for i in combined_reviews.split():

    list1.append(i)

dictionary1 = {}

for j in list1:

    dictionary1[j] = dictionary1.get(j,0)+1

series1 = pd.Series(dictionary1)

word_freq = pd.DataFrame(series1)

word_freq = word_freq.reset_index().rename(columns = {'index':'Words', 0:'Frequency'})

word_freq.head()
top_25_words = word_freq.sort_values(ascending = False, by = 'Frequency')

top_25_words.head(25)
last_25_words = word_freq.sort_values(ascending = False, by = 'Frequency')

last_25_words.tail(25)
from nltk.tokenize import word_tokenize

all_words = word_tokenize(combined_reviews.lower())

print(all_words[:200])
from nltk.probability import FreqDist

fdist = FreqDist(all_words)

fdist
plt.figure(figsize = (10,6))

fdist.plot(25, cumulative = False)

plt.show()
from nltk.corpus import stopwords

from string import punctuation

stop_words = stopwords.words('english')

print(stop_words)

print(list(punctuation))
stop_words_updated = stop_words + ['..', '...', 'will', 'would', 'can', 'could', "n't"]

print(stop_words_updated)
all_words_updated = [word for word in all_words if word not in stop_words_updated\

                  and word not in list(punctuation) and len(word) > 2]

print(all_words_updated[:200])
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

lemma_words = [lemma.lemmatize(i) for i in all_words_updated]

print(len(set(lemma_words)))
def clean_text(text):

    token = word_tokenize(text.lower())

    lemm = [lemma.lemmatize(i) for i in token if i not in stop_words_updated\

           and i not in list(punctuation) and len(i) > 2]

    sentence = ' '.join(lemm)

    return sentence



reviews['clean_reviewText'] = reviews['reviewText'].apply(clean_text)

reviews.head()
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(ngram_range = (2,2))

bigrams = count_vect.fit_transform(reviews['clean_reviewText'])

print(count_vect.get_feature_names()[:200])
DTM = pd.DataFrame(bigrams.toarray(), columns = count_vect.get_feature_names())

DTM.head()
top_25_bigrams = DTM.sum().sort_values(ascending = False).head(25)

top_25_bigrams
top_25_bigrams.plot(kind = 'bar', figsize = (16,8))

plt.show()
bigrams = DTM.columns

print(bigrams)
negative_words = ['poor', 'waste', 'bad', 'defective', 

                  'disgusting', 'untrusty', 'worst', 

                  'horrible', 'unexpectedly', 'slow']
negative_bigrams = []

for i in bigrams:

    words = i.split()

    if sum(np.in1d(words, negative_words)) >= 1:

        negative_bigrams.append(i)
DTM_subset = DTM[negative_bigrams]

top_25_cutomer_concern_areas = DTM_subset.sum().sort_values(ascending = False).head(25)

top_25_cutomer_concern_areas
top_25_cutomer_concern_areas.plot(kind = 'bar', figsize = (16,8))