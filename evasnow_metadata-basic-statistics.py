import numpy as np

import pandas as pd

import os

import collections

import re

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer

import time

import spacy

import string
metaDataPath = "/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv"

metaData = pd.read_csv(metaDataPath, header = 0, index_col = 0)

print("The number of literatures: " + str(metaData.shape[0]))

metaData.head()
metaData.isnull().sum()
sourceDic = collections.defaultdict(int)

for s in metaData["source_x"][metaData["source_x"].notnull()]:

    sourceDic[s] += 1

sizes = []

explode = []

labels = []

for s in sourceDic:

    sizes.append(sourceDic[s])

    explode.append(0)

    labels.append(s)
colors = ['gold', 'lightskyblue', 'yellowgreen', 'lightcoral']

plt.pie(sizes, explode=explode, labels=labels, colors = colors, autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.show()
yearList = []

for y in metaData["publish_time"][metaData["publish_time"].notnull()]:

    yearList.append(int(re.split(' |-', y)[0]))
sns.distplot(yearList, bins = 50)

plt.xlabel("Year")

plt.ylabel("Frequency")
hasFullText = metaData["has_full_text"][metaData["has_full_text"].notnull()]

nanCount = metaData.shape[0] - hasFullText.shape[0]

trueCount = sum(hasFullText)

falseCount = hasFullText.shape[0] - trueCount

print("The number of literatures with full text: " + str(trueCount))

print("The number of literatures without full text: " + str(falseCount + nanCount))
print("The number of literatures with abstract: " + str(sum(metaData["abstract"].notnull())))
startTime = time.time()

absLength = []

word2count = {}

for abstract in metaData["abstract"][metaData["abstract"].notnull()]:

    ## Remove web links

    abstract = re.sub('https?://\S+|www\.\S+', '', abstract) 



    ## Lowercase

    abstract = abstract.lower()

    

    ## Remove punctuation

    abstract = re.sub('<.*?>+', ' ', abstract)

    abstract = re.sub('[%s]' % re.escape(string.punctuation), ' ', abstract)

    

    ## Tokenize

    words = word_tokenize(abstract)

    

    ## Remove stop words

    nltk_stop_words = stopwords.words('english')

    words = [word for word in words if word not in nltk_stop_words]

    

    ## Stem

    stemmer = SnowballStemmer('english')

    words = [stemmer.stem(word) for word in words]

    

    ## Lematize verbs

    lemmatizer = WordNetLemmatizer()

    words = [lemmatizer.lemmatize(word, pos='v') for word in words]

    

    ## Record length

    absLength.append(len(words))

    

    ## Get word count

    for word in words:

        count = word2count.get(word, 0)

        word2count[word] = count + 1

print("Time spent: " + str(round((time.time() - startTime) / 60, 3)) + "min.")
sns.distplot(sorted(absLength)[:-20], bins = 50) # There are 20 extremely long abstracts

plt.xlabel("Abstract token count")

plt.ylabel("Frequency")

plt.show()
df_word_count = pd.DataFrame(sorted(word2count.items(), key=lambda x: x[1])[::-1])

sns.set(rc={'figure.figsize':(12,10)})

sns.barplot(y = df_word_count[0].values[:50], x = df_word_count[1].values[:50], color='red')