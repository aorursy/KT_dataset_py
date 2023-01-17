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
#First of all, I need to import the following libraries:

## for data

import json

import pandas as pd

import numpy as np

## for plotting

import matplotlib.pyplot as plt

import seaborn as sns

## for processing

import re

import nltk

## for bag-of-words

from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing

## for explainer

from lime import lime_text

## for word embedding

import gensim

import gensim.downloader as gensim_api

## for deep learning

from tensorflow.keras import models, layers, preprocessing as kprocessing

from tensorflow.keras import backend as K

## for bert language model

import transformers
dtf = pd.read_csv("../input/nlp-getting-started/train.csv")
fig, ax = plt.subplots()

fig.suptitle("target", fontsize=12)

dtf["target"].reset_index().groupby("target").count().sort_values(by= "index").plot(kind="barh", legend=False, 

        ax=ax).grid(axis='x')

plt.show() 
'''

Preprocess a string.

:parameter

    :param text: string - name of column containing text

    :param lst_stopwords: list - list of stopwords to remove

    :param flg_stemm: bool - whether stemming is to be applied

    :param flg_lemm: bool - whether lemmitisation is to be applied

:return

    cleaned text

'''

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):

    ## clean (convert to lowercase and remove punctuations and   

    #characters and then strip

    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

            

    ## Tokenize (convert from string to list)

    lst_text = text.split()

    ## remove Stopwords

    if lst_stopwords is not None:

        lst_text = [word for word in lst_text if word not in 

                    lst_stopwords]

                

    ## Stemming (remove -ing, -ly, ...)

    if flg_stemm == True:

        ps = nltk.stem.porter.PorterStemmer()

        lst_text = [ps.stem(word) for word in lst_text]

                

    ## Lemmatisation (convert the word into root word)

    if flg_lemm == True:

        lem = nltk.stem.wordnet.WordNetLemmatizer()

        lst_text = [lem.lemmatize(word) for word in lst_text]

            

    ## back to string from list

    text = " ".join(lst_text)

    return text
lst_stopwords = nltk.corpus.stopwords.words("english")

lst_stopwords
dtf["text_clean"] = dtf["text"].apply(lambda x: 

          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 

          lst_stopwords=lst_stopwords))

dtf.head()
## split dataset

dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)

## get target

y_train = dtf_train["target"].values

y_test = dtf_test["target"].values
## Count (classic BoW)

vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))



## Tf-Idf (advanced variant of BoW)

vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))
corpus = dtf_train["text_clean"]

vectorizer.fit(corpus)

X_train = vectorizer.transform(corpus)

dic_vocabulary = vectorizer.vocabulary_
word = "forest"

dic_vocabulary[word]

#If the word exists in the vocabulary, 

#this command prints a number N, 

#meaning that the Nth feature of the matrix is that word.
from sklearn import feature_selection 

y = dtf_train["target"]

X_names = vectorizer.get_feature_names()

p_value_limit = 0.95

dtf_features = pd.DataFrame()

for cat in np.unique(y):

    chi2, p = feature_selection.chi2(X_train, y==cat)

    dtf_features = dtf_features.append(pd.DataFrame(

                   {"feature":X_names, "score":1-p, "y":cat}))

    dtf_features = dtf_features.sort_values(["y","score"], 

                    ascending=[True,False])

    dtf_features = dtf_features[dtf_features["score"]>p_value_limit]

X_names = dtf_features["feature"].unique().tolist()

len(X_names)
for cat in np.unique(y):

    print("# {}:".format(cat))

    print("  . selected features:",

         len(dtf_features[dtf_features["y"]==cat]))

    print("  . top features:", ",".join(

dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))

    print(" ")
classifier = naive_bayes.MultinomialNB()
## pipeline

model = pipeline.Pipeline([("vectorizer", vectorizer),  

                           ("classifier", classifier)])

## train classifier

model["classifier"].fit(X_train, y_train)

## test

X_test = dtf_test["text_clean"].values

predicted = model.predict(X_test)

predicted_prob = model.predict_proba(X_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, predicted)

accuracy 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(classification_report(y_test, predicted))
# Generate confusion matrix

from sklearn.metrics import plot_confusion_matrix

from mlxtend.plotting import plot_decision_regions

matrix = plot_confusion_matrix(model, X_test, y_test,

                                 cmap=plt.cm.Blues,

                                 normalize='true')

plt.title('Confusion matrix for our classifier')

plt.show(matrix)

plt.show()