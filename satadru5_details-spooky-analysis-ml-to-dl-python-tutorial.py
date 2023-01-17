# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Import python libraries 

import base64

import pandas as pd

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from collections import Counter

from scipy.misc import imread

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head(5)
test.head(5)
sns.countplot(train['author'])
eap = train[train.author=="EAP"]["text"].values

hpl = train[train.author=="HPL"]["text"].values

mws = train[train.author=="MWS"]["text"].values
dic={'EAP':1,'HPL':2,'MWS':3}

train['author']=train.author.map(dic)
from wordcloud import WordCloud, STOPWORDS

import nltk

from nltk.corpus import stopwords
# The wordcloud of Cthulhu/squidy thing for HP Lovecraft

plt.figure(figsize=(14,11))

wc = WordCloud(background_color="black", max_words=10000, 

                stopwords=STOPWORDS, max_font_size= 25)

wc.generate(" ".join(hpl))

plt.title("HP Lovecraft (Cthulhu-Squidy)", fontsize=16)

plt.imshow(wc.recolor( colormap= 'cubehelix_r' , random_state=17), alpha=0.9)
plt.figure(figsize=(14,11))

wc = WordCloud(background_color="black", max_words=10000, 

                stopwords=STOPWORDS, max_font_size= 25)

wc.generate(" ".join(eap))

plt.title("EAP - Edgar Allen Poe", fontsize=16)

plt.imshow(wc.recolor( colormap= 'cubehelix_r' , random_state=17), alpha=0.9)
plt.figure(figsize=(14,11))

wc = WordCloud(background_color="black", max_words=10000, 

                stopwords=STOPWORDS, max_font_size= 25)

wc.generate(" ".join(mws))

plt.title("MWS - Mary Shelley", fontsize=16)

plt.imshow(wc.recolor( colormap= 'cubehelix_r' , random_state=17), alpha=0.9)
import nltk
first_text = train.text.values[0]

first_text_list = nltk.word_tokenize(first_text)

print(first_text_list)
stopwords = nltk.corpus.stopwords.words('english')

first_text_list_cleaned = [word for word in first_text_list if word.lower() not in stopwords]

print(first_text_list_cleaned)

print("="*90)

print("Length of original list: {0} words\n"

      "Length of list after stopwords removal: {1} words"

      .format(len(first_text_list), len(first_text_list_cleaned)))
from nltk.corpus import stopwords

stop = stopwords.words('english')

train['text_without_stopwords'] = train['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

test['text_without_stopwords'] = test['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train.head()
stemmer = nltk.stem.PorterStemmer()

print(stemmer.stem("playing"))

print(stemmer.stem("play"))
from nltk.corpus import stopwords

stop = stopwords.words('english')

train['text_without_stopwords'] = train['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

test['text_without_stopwords'] = test['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
import nltk.stem as stm # Import stem class from nltk

import re

stemmer = stm.PorterStemmer()



# Crazy one-liner code here...

# Explanation above...

train.text_without_stopwords = train.text_without_stopwords.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))

test.text_without_stopwords = test.text_without_stopwords.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))
def tokenize_and_stem(text):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token

    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    filtered_tokens = []

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)

    for token in tokens:

        if re.search('[a-zA-Z]', token):

            filtered_tokens.append(token)

    stems = [stemmer.stem(t) for t in filtered_tokens]

    return stems
from sklearn.feature_extraction.text import CountVectorizer # Import the library to vectorize the text



# Instantiate the count vectorizer with an NGram Range from 1 to 3 and english for stop words.

count_vect = CountVectorizer(ngram_range=(1,3),stop_words='english')



# Fit the text and transform it into a vector. This will return a sparse matrix.

count_vectorized_train = count_vect.fit_transform(train.text_without_stopwords) #for train data set

count_vectorized_test = count_vect.fit_transform(test.text_without_stopwords) #for test data set
def tokenize_and_stem(text):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token

    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    filtered_tokens = []

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)

    for token in tokens:

        if re.search('[a-zA-Z]', token):

            filtered_tokens.append(token)

    stems = [stemmer.stem(t) for t in filtered_tokens]

    return stems
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer # Import the library to vectorize the text

tfidf_vect = TfidfVectorizer(stop_words='english',

                                 use_idf=True, tokenizer=tokenize_and_stem)#We have used this as default model.

#we can tune this to get more accuracy

#tfidf_vect = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

#This two our test and training dataset

tfidf_vectorized_train = tfidf_vect.fit_transform(train.text)

tfidf_vectorized_test = tfidf_vect.fit_transform(test.text)
from sklearn.model_selection import train_test_split # Import the function that makes splitting easier.



# Split the vectorized data. Here we pass the vectorized values and the author column.

# Also, we specify that we want to use a 75% of the data for train, and the rest for test.



###########################

# COUNT VECTORIZED TOKENS #

###########################

X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(count_vectorized_train, train.author, train_size=0.75)



###########################

# TFIDF VECTORIZED TOKENS #

###########################

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_vectorized_train, train.author, train_size=0.75)
# First, import the Multinomial Naive bayes library from sklearn 

from sklearn.naive_bayes import MultinomialNB



# Instantiate the model.

# One for Count Vectorized words

model_count_NB = MultinomialNB()

# One for TfIdf vectorized words

model_tfidf_NB = MultinomialNB()



# Train the model, passing the x values, and the target (y)

model_count_NB.fit(X_train_count, y_train_count)

model_tfidf_NB.fit(X_train_tfidf, y_train_tfidf)
# Predict the values, using the test features for both vectorized data.

predictions_count = model_count_NB.predict(X_test_count)

predictions_tfidf = model_tfidf_NB.predict(X_test_tfidf)
# Primero calculamos el accuracy general del modelo

from sklearn.metrics import accuracy_score

accuracy_count = accuracy_score(y_test_count, predictions_count)

accuracy_tfidf = accuracy_score(y_test_tfidf, predictions_tfidf)

print('Count Vectorized Words Accuracy:', accuracy_count)

print('TfIdf Vectorized Words Accuracy:', accuracy_tfidf)
from sklearn.neighbors import KNeighborsClassifier

model_count_kmn = KNeighborsClassifier(n_neighbors=10)

model_tfidf_kmn = KNeighborsClassifier(n_neighbors=10)

model_count_kmn.fit(X_train_count, y_train_count)

model_tfidf_kmn.fit(X_train_tfidf, y_train_tfidf)

predictions_count_kmn = model_count_kmn.predict(X_test_count)

predictions_tfidf_kmn = model_tfidf_kmn.predict(X_test_tfidf)

accuracy_count_kmn = accuracy_score(y_test_count, predictions_count_kmn)

accuracy_tfidf_kmn = accuracy_score(y_test_tfidf, predictions_tfidf_kmn)

print('Count Vectorized Words Accuracy:', accuracy_count_kmn)

print('TfIdf Vectorized Words Accuracy:', accuracy_tfidf_kmn)

from sklearn.ensemble import RandomForestClassifier

model_rf_count = RandomForestClassifier()

model_rf_tfidf = RandomForestClassifier()

model_rf_count.fit(X_train_count, y_train_count)

model_rf_tfidf.fit(X_train_tfidf, y_train_tfidf)

predictions_count_rf = model_rf_count.predict(X_test_count)

predictions_tfidf_rf = model_rf_tfidf.predict(X_test_tfidf)

accuracy_count_rf = accuracy_score(y_test_count, predictions_count_rf)

accuracy_tfidf_rf = accuracy_score(y_test_tfidf, predictions_tfidf_rf)

print('Count Vectorized Words Accuracy:', accuracy_count_rf)

print('TfIdf Vectorized Words Accuracy:', accuracy_tfidf_rf)
from sklearn.grid_search import GridSearchCV  

from time import time  ;

from sklearn.pipeline import Pipeline  
#test train split once again for pipeline and greadsearch

X_train_gd, X_test_gd, y_train_gd, y_test_gd = train_test_split(train.text_without_stopwords, train.author, train_size=0.75)

pipeline = Pipeline([('vec', count_vect),('tfidf', TfidfTransformer()),

 ('clf',MultinomialNB())])
parameters = {  

    'vec__max_df': (0.5, 0.625, 1.0),  

    'vec__max_features': (None, 5000),  

    'vec__min_df': (1, 5, 10),  

    'tfidf__use_idf': (True, False)  

 

    }
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2)  

t0 = time()  

grid_search.fit(X_train_gd, y_train_gd)  

print("done in {0}s".format(time() - t0))  

print("Best score: {0}".format(grid_search.best_score_))  

print("Best parameters set:")  

best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(list(parameters.keys())):  

            print("\t{0}: {1}".format(param_name, best_parameters[param_name]))
#need to create function for all clssifier#

#lgboost and xgboost need to be done

#sumission from best classifier

#ploting accuracy

#Deep learning