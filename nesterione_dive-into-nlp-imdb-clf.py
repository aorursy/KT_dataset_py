# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# configure matplotlib for charts 

%matplotlib inline 

import matplotlib.pyplot as plt



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import re # regex 



import nltk # 

import numpy as np

import pandas as pd

from nltk.corpus import stopwords # stop words

from nltk.stem import WordNetLemmatizer # lemmatizer 

from sklearn.decomposition import TruncatedSVD # dimensionality reduction for charts

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # text vectorization algorithms 

from sklearn.linear_model import LogisticRegression # classification algorithm 

from sklearn.metrics import classification_report, confusion_matrix #colculation metrix and show results



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load IMDB dataset

dataset = pd.read_csv('../input/imdb_master.csv', encoding='windows-1252')

dataset.head()



train = dataset[dataset.type == 'train']

test = dataset[dataset.type == 'test']

(train_texts, train_labels),(test_texts, test_labels) = (train.review, train.label), (test.review, test.label)



train_texts = train_texts[train_labels != 'unsup']

train_labels = train_labels[train_labels != 'unsup']

test_texts = test_texts[test_labels != 'unsup']

test_labels = test_labels[test_labels != 'unsup']



train_labels[train_labels=='pos'] = 1

train_labels[train_labels=='neg'] = 0

train_labels = train_labels.astype(np.int)

test_labels[test_labels=='pos'] = 1

test_labels[test_labels=='neg'] = 0

test_labels = test_labels.astype(np.int)



print("train labels", set(train_labels))

print("test labels", set(test_labels))
# Show some examples

train_texts[25000],'', train_texts[25234],'', train_texts[29366]
# print with default NLTK tokenization

print(nltk.word_tokenize(train_texts[25000]))
# Define cleaning function 

# NOTE: try use different clearning techniques

pattern = re.compile(r"[^a-zA-Z ]+")



def preproc(text):

    return pattern.sub('', text.lower())
print(nltk.word_tokenize(preproc(train_texts[25000])))
# Defining normalization functions  



# lemmatizator WordNet

stemmer = WordNetLemmatizer()



# stop words for english 

stop_words = stopwords.words('english')



def nornalize(words, use_lemma = False):

    filtered_sentence = [w for w in words if not w in stop_words] 

    

    if use_lemma:

        filtered_sentence = [ stemmer.lemmatize(w) for w in filtered_sentence]

        

    return filtered_sentence



def tokenize(text, use_lemma=False):

    words = nltk.word_tokenize(preproc(text))

    return nornalize(words, use_lemma)
# Show examples of tokenization 



# without lemmatization

print(tokenize(train_texts[25000]))



# with lemmatization 

print(tokenize(train_texts[25000], use_lemma=True))
# Define function to plot chart 



def plot_vectors(vectors, labels):

    y_chart_neg = labels == 0

    y_chart_pos = labels == 1

    

    # make dimensionality reduction to whow on 2D chart

    svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)

    X_chart = svd.fit_transform(vectors)  

    

    plt.figure(figsize=(10, 10))

    plt.scatter(X_chart[y_chart_pos,0], X_chart[y_chart_pos,1], marker='o', c='b')

    plt.scatter(X_chart[y_chart_neg,0], X_chart[y_chart_neg,1], marker='x', c='r')

    plt.show()
%%time

bow_vectorizer = CountVectorizer(preprocessor=preproc, tokenizer=(lambda t: tokenize(t)), ngram_range=(1, 1), max_features = 60000)

bow_train = bow_vectorizer.fit_transform(train_texts)

bow_train.shape
plot_vectors(bow_train, train_labels)
# Train classification algorithm with default values 



lr = LogisticRegression()

lr.fit(bow_train, train_labels)



# predict values

y_pred = lr.predict(bow_train)
# analyse results with training set 



print(classification_report(train_labels, y_pred))

print(confusion_matrix(train_labels, y_pred))
# validate on test set 
%%time

bow_test = bow_vectorizer.transform(test_texts)
y_pred_test = lr.predict(bow_test)
print(classification_report(test_labels, y_pred_test))

print(confusion_matrix(test_labels, y_pred_test))
%%time

tfidf_vectorizer = TfidfVectorizer(preprocessor=preproc, tokenizer=(lambda t: tokenize(t)), ngram_range=(1, 2), max_features = 60000 )

tfidf_train = tfidf_vectorizer.fit_transform(train_texts)

plot_vectors(tfidf_train, train_labels)
# Train classification algorithm with default values 

lr = LogisticRegression()

lr.fit(tfidf_train, train_labels)

# analyse results with training set 

y_pred = lr.predict(tfidf_train)

print(classification_report(train_labels, y_pred))

print(confusion_matrix(train_labels, y_pred))
# validate on test set 

tfidf_test = tfidf_vectorizer.transform(test_texts)
y_test_pred = lr.predict(tfidf_test)



print(classification_report(test_labels, y_test_pred))

print(confusion_matrix(test_labels, y_test_pred))