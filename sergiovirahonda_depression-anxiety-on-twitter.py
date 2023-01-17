from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import re

import numpy as np

import pandas as pd 

import os

import matplotlib.pyplot as plt

import string

from nltk.corpus import stopwords

import nltk

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

from collections import Counter

from wordcloud import WordCloud

from nltk.corpus import stopwords

import nltk

from gensim.utils import simple_preprocess

from nltk.corpus import stopwords

import gensim

from sklearn.model_selection import train_test_split

import spacy

from sklearn.decomposition import NMF, LatentDirichletAllocation

import pyLDAvis

import pyLDAvis.sklearn

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from pprint import pprint

from time import time

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb

from sklearn.metrics import precision_score, recall_score, accuracy_score,roc_auc_score

import warnings

warnings.filterwarnings('ignore')

from datetime import datetime

import seaborn as sns

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import keras

print('Done')
os.listdir('/kaggle/input')
dataset = pd.read_excel('../input/depression-and-anxiety-comments/Depression  Anxiety Facebook page Comments Text.xlsx')

dataset.head()
dataset.shape
dataset.isnull().sum()
#Removing URLs with a regular expression



def remove_urls(text):

    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    return url_pattern.sub(r'', text)



for i in range(len(dataset)):

  dataset.at[i,'Comments Text'] = remove_urls(dataset.iloc[i]['Comments Text'])

dataset.head()
# Convert to list

data = dataset['Comments Text'].values.tolist()



# Remove Emails

data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]



# Remove new line characters

data = [re.sub('\s+', ' ', sent) for sent in data]



# Remove distracting single quotes

data = [re.sub("\'", "", sent) for sent in data]



print(data[:1])
def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



data_words = list(sent_to_words(data))



print(data_words[:1])
# Build the bigram and trigram models

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  



# Faster way to get a sentence clubbed as a trigram/bigram

bigram_mod = gensim.models.phrases.Phraser(bigram)

trigram_mod = gensim.models.phrases.Phraser(trigram)



# See trigram example

print(trigram_mod[bigram_mod[data_words[0]]])
# Define functions for stopwords, bigrams, trigrams and lemmatization



stop_words = set(stopwords.words("english"))





def remove_stopwords(texts):

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



def make_bigrams(texts):

    return [bigram_mod[doc] for doc in texts]



def make_trigrams(texts):

    return [trigram_mod[bigram_mod[doc]] for doc in texts]



def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    texts_out = []

    for sent in texts:

        doc = nlp(" ".join(sent)) 

        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out
# Remove Stop Words

data_words_nostops = remove_stopwords(data_words)



# Form Bigrams

data_words_bigrams = make_bigrams(data_words_nostops)



# Initialize spacy 'en' model, keeping only tagger component (for efficiency)

# python3 -m spacy download en

nlp = spacy.load('en', disable=['parser', 'ner'])



# Do lemmatization keeping only noun, adj, vb, adv

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])



print(data_lemmatized[:1])
dataset = []

for i in range(len(data_lemmatized)):

    dataset.append(" ".join(data_lemmatized[i]))

dataset = pd.Series(dataset)
no_features = 15000



# NMF is able to use tf-idf

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=no_features)

tfidf = tfidf_vectorizer.fit_transform(dataset)

tfidf_feature_names = tfidf_vectorizer.get_feature_names()



# LDA can only use raw term counts for LDA because it is a probabilistic graphical model

tf_vectorizer = CountVectorizer(min_df=0.05,max_features=no_features)

tf = tf_vectorizer.fit_transform(dataset)

tf_feature_names = tf_vectorizer.get_feature_names()
no_topics = 2



# Run NMF

nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5,max_iter=10000).fit(tfidf)



# Run LDA

lda = LatentDirichletAllocation(n_components=no_topics, max_iter=10, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
def display_topics(model, feature_names, no_top_words):

    for topic_idx, topic in enumerate(model.components_):

        print("Topic %d:" % (topic_idx))

        print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))



no_top_words = 25

print('NMF')

display_topics(nmf, tfidf_feature_names, no_top_words)

print('LDA')

display_topics(lda, tf_feature_names, no_top_words)
# Create Document — Topic Matrix

lda_output = lda.transform(tf)

# column names

topicnames = ['Topic' + str(i) for i in range(lda.n_components)]

# index names

docnames = ['Doc' + str(i) for i in range(len(dataset))]

# Make the pandas dataframe

df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document

dominant_topic = np.argmax(df_document_topic.values, axis=1)

df_document_topic['dominant_topic'] = dominant_topic



df_document_topics = df_document_topic

dataset2 = pd.read_excel('../input/depression-and-anxiety-comments/Depression  Anxiety Facebook page Comments Text.xlsx')

df_document_topics.reset_index(inplace=True,drop=True)

dataset2['label'] = df_document_topics['dominant_topic']
dataset2.head()
# Create Document — Topic Matrix

nmf_output = nmf.transform(tfidf)

# column names

topicnames = ['Topic' + str(i) for i in range(nmf.n_components)]

# index names

docnames = ['Doc' + str(i) for i in range(len(dataset))]

# Make the pandas dataframe

df_document_topic = pd.DataFrame(np.round(nmf_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document

dominant_topic = np.argmax(df_document_topic.values, axis=1)

df_document_topic['dominant_topic'] = dominant_topic



df_document_topics = df_document_topic

dataset1 = pd.read_excel('../input/depression-and-anxiety-comments/Depression  Anxiety Facebook page Comments Text.xlsx')

df_document_topics.reset_index(inplace=True,drop=True)

dataset1['label'] = df_document_topics['dominant_topic']
dataset1.head()
dataset1[dataset1['label']==1]
for i in range(20):

    print(dataset1[dataset1['label']==1].iloc[i][0])

    print('\n')
dataset2[dataset2['label']==1]
for i in range(20):

    print(dataset2[dataset2['label']==1].iloc[i][0])

    print('\n')
dataset1.head(15)
for i in range(len(dataset2)):

  dataset1.at[i,'Comments Text'] = remove_urls(dataset1.iloc[i]['Comments Text'])

dataset1.head()
# Convert to list

data = dataset1['Comments Text'].values.tolist()



# Remove Emails

data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]



# Remove new line characters

data = [re.sub('\s+', ' ', sent) for sent in data]



# Remove distracting single quotes

data = [re.sub("\'", "", sent) for sent in data]



# Remove distracting commas

data = [re.sub(",", "", sent) for sent in data]



# Remove distracting commas

data = [sent.lower() for sent in data]



# Remove distracting dots

data = [sent.replace('.', '') for sent in data]



print(data[:1])
tweets = np.array(data)

labels = np.array(dataset2['label'])
print(len(tweets),len(labels))
from keras.models import Sequential

from keras import layers

from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras import regularizers

max_words = 20000

max_len = 400



tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(tweets)

sequences = tokenizer.texts_to_sequences(tweets)

tweets = pad_sequences(sequences, maxlen=max_len)

print(tweets)
X_train, X_test, y_train, y_test = train_test_split(tweets,labels, random_state=0)

print (len(X_train),len(X_test),len(y_train),len(y_test))
model1 = Sequential()

model1.add(layers.Embedding(max_words, 40))

model1.add(layers.LSTM(40,dropout=0.5))

model1.add(layers.Dense(1,activation='sigmoid'))



model1.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])



history = model1.fit(X_train, y_train, epochs=7,validation_data=(X_test, y_test))
test_loss, test_acc = model1.evaluate(X_test,  y_test, verbose=2)

print('Model accuracy: ',test_acc)
y_pred = model1.predict(X_test)
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, np.around(y_pred, decimals=0))

import seaborn as sns

conf_matrix = pd.DataFrame(matrix, index = ['Not Depression/Anxiety','Anxiety/Depression'],columns = ['Not Depression/Anxiety','Anxiety/Depression'])

#Normalizing

conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize = (15,15))

sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})
model2 = Sequential()

model2.add(layers.Embedding(max_words, 40))

model2.add(layers.LSTM(40,dropout=0.5,return_sequences=True))

model2.add(layers.LSTM(40,dropout=0.5))

model2.add(layers.Dense(1,activation='sigmoid'))



model2.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])



history = model2.fit(X_train, y_train, epochs=5,validation_data=(X_test, y_test))
test_loss, test_acc = model2.evaluate(X_test,  y_test, verbose=2)

print('Model accuracy: ',test_acc)
y_pred = model2.predict(X_test)
matrix = confusion_matrix(y_test, np.around(y_pred, decimals=0))

conf_matrix = pd.DataFrame(matrix, index = ['Not Depression/Anxiety','Anxiety/Depression'],columns = ['Not Depression/Anxiety','Anxiety/Depression'])

#Normalizing

conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize = (15,15))

sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})
model3 = Sequential()

model3.add(layers.Embedding(max_words, 40))

model3.add(layers.Bidirectional(layers.LSTM(40,dropout=0.5)))

model3.add(layers.Dense(1,activation='sigmoid'))



model3.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])



history = model3.fit(X_train, y_train, epochs=8,validation_data=(X_test, y_test))
y_pred = model3.predict(X_test)
matrix = confusion_matrix(y_test, np.around(y_pred, decimals=0))

conf_matrix = pd.DataFrame(matrix, index = ['Not Depression/Anxiety','Anxiety/Depression'],columns = ['Not Depression/Anxiety','Anxiety/Depression'])

#Normalizing

conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize = (15,15))

sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})
test = np.array(['I feel stress, sadness and anxiety - just want to sleep until the lockdown ends'])

test_sequence = tokenizer.texts_to_sequences(test)

test_sequence = pad_sequences(test_sequence, maxlen=max_len)

test_prediction = model3.predict(test_sequence)

if np.around(test_prediction, decimals=0)[0][0] == 1.0:

    print('The model predicted depressive/anxious language')

else:

    print("The model predicted other type of language")
os.listdir('/kaggle/input/depression-anxiety-tweets/Tweets data')[:5]
tweets = pd.read_csv('/kaggle/input/depression-anxiety-tweets/Tweets data/0314_1.csv')

tweets.head()
for dirname, _, filenames in os.walk('/kaggle/input/depression-anxiety-tweets/Tweets data'):

    for filename in filenames:

        if filename!='0314_1.csv':

            temp = pd.read_csv(os.path.join(dirname, filename))

            tweets = pd.concat([tweets, temp], ignore_index=True)
tweets.shape
tweets.sort_values(by=['date'],inplace=True)

tweets.reset_index(drop=True,inplace=True)

tweets = tweets[['date','text']]
tweets_dataset = tweets.copy()

tweets.head()
#Removing non-ascii characters (for example, arabian chars)

tweets.text.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)

#Making all fields string type

for i in range(len(tweets)):

  tweets.at[i,'text'] = str(tweets.iloc[i]['text'])

#Removing URLs

for i in range(len(tweets)):

  tweets.at[i,'text'] = remove_urls(tweets.iloc[i]['text'])

# Convert to list

data = tweets.text.values.tolist()

# Remove Emails

data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters

data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes

data = [re.sub("\'", "", sent) for sent in data]
data = np.array(data)

data[:10]
sequences = tokenizer.texts_to_sequences(data)

tweets = pad_sequences(sequences, maxlen=max_len)

print(tweets)
predictions = model3.predict(tweets)
np.around(predictions, decimals=0)
tweets_dataset['label'] = np.around(predictions, decimals=0)
tweets_dataset[tweets_dataset['label']==1.0].head(10)
for i in range(10):

    print(tweets_dataset.iloc[i*2]['text'])

    print('\n')