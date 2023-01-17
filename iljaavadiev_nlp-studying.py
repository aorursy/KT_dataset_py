import numpy as np 

import pandas as pd 

pd.set_option('display.max_colwidth', None)

#pd.set_option('display.max_rows', 0)



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

import re



#model selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split



#nlp libraries

import spacy

from spacy import displacy

from sklearn.feature_extraction.text import TfidfVectorizer



#ml

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from sklearn.neighbors import KNeighborsClassifier





#metrics

from sklearn.metrics import f1_score





#deep learning

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.text import Tokenizer
device = 'CPU'

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if tf.config.experimental.list_physical_devices('GPU'):

    device = 'GPU'
#print the current library versions

packages = [np, pd, mpl, sns, spacy]



print('-'*35)

for package in packages:

    print(package.__name__, 'version:', package.__version__)

    print('-'*35)
paths = []

for root, dirs, files in os.walk('/kaggle/input'):

    for file in sorted(files):

        paths.append(os.path.join(root, file))

print(paths)
train_df = pd.read_csv(paths[2], index_col='id')

test_df = pd.read_csv(paths[1], index_col='id')

concat = [train_df, test_df]
# how large is the corpus

print('CORPUS')

print('Training set corpus size:', train_df.shape[0])

print('Test set corpus size:', test_df.shape[0])
# keyword and location are apparently missing in many cases

train_df.head()
# we will have to deal with keyword and location null values

print('-'*50)

print('\nTRAINING DATA:\n')

print(train_df.info())

print('-'*50)

print('\nTESTING DATA:\n')

print(test_df.info())
#how do keywords look like?

#there are 221 unique values

train_df['keyword'].value_counts()
# how do locations look like?

# there are a lot unique values, which probably don't have a lot of predictive power

train_df['location'].value_counts()
# lets look at some regex examples

example = 'The quick brown fox jumps over the lazy dog.'

# find the word jumps

print(re.findall(r'j\w+', example))

# find words that are exactly 4 letters long

print(re.findall(r'\b\w{4}\b', example))

# find pairs of words

print(re.findall(r'\w+\s\w+', example))

# find the last word in the sentence

print(re.findall(r'\w+.?$', example))
for df in concat:  

    # find a hashtag and create a new column

    df['hashtags'] = df.text.str.findall(r'#\w+')

    # find a user mentioned and create a new column

    df['user'] = df.text.str.findall(r'@\w*')
df.head()
# how many hashtags are in a tweet?

for df in concat:

    df['hash_count'] = df['hashtags'].str.len()
df.head()
# most tweets don't actually have a hashtag

plt.figure(figsize=(20,10))

sns.countplot(x='hash_count', data=train_df).set_title('Count for number of #hashtags')

plt.show()
# it does not look like people necessarily use more hashtags in case of emmergency

plt.figure(figsize=(20,10))

sns.barplot(x='hash_count', y='target', data=train_df).set_title('Number of hashtags vs % of real disaster tweets')

plt.show()
#how many of the tweets are actual disater tweets

sns.countplot(x='target', data=train_df).set_title('Not Disaster vs Disater Tweets')

plt.show()
#look at the tweets themselves

#some of the text seems to repeat

train_df.describe(include=object)
for df in concat:

    df['length'] = pd.Series(df.loc[:, 'text'].str.len())

    df['length'] = pd.Series(df.loc[:, 'text'].str.len())
plt.figure(figsize=(25, 10))

axes = sns.countplot(x='length', data=train_df)

axes.set_title('Length of tweet and disaster')

axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

plt.show()
#deal with missing hashtags

for df in concat:

    df['hashtags'] = df['hashtags'].apply(lambda x: ['NoHashTag'] if not x else x)

    df['hashtags'] = df['hashtags'].apply(lambda x: ' '.join(x))
#deal with missing location and keyword

for df in concat:

    df['keyword'].fillna(value='Missing', inplace=True)

    df['location'].fillna(value='Missing', inplace=True)
test_df.head()
#clean text

#lets write a function, as we probably have to do several cleaning steps

#there are probably better solutions out there

#TODO look at other notebooks

def clean_tweet(doc):

    

    #remove urls

    doc =  re.sub(r'https?://\w?\.?\w*/\w*', '', doc)

    #remove @user

    doc = re.sub(r'@\w*', '', doc)

    #remove date in the format nn/nn/nn

    doc = re.sub(r'\d+/\d+/\d+', '', doc)

    #remove time in the format hh:mm

    doc = re.sub(r'\d+:\d+', '', doc)

    #remove special signs

    doc = re.sub(r'[#@.?:-=/\\<>\]\[]', '', doc)

    #remove words containing numbers

    doc = re.sub(r'(\w+\d+|\d+\w+)', '', doc)

    return doc



for df in concat:

    df['text'] = df['text'].apply(clean_tweet)

train_df.head()
# 1-gram tokenizer

example = 'The quick brown fox jumps over the lazy dog.'



# remove the dots and make all words lower case

clean_example = re.sub(r'\.', '', example).lower()

print(clean_example.split())
# 2-gram tokenizer



example = 'The quick brown fox jumps over the lazy dog.'



without_first = example.split()[1:]

without_last = example.split()[:-1]



list(zip(without_last, without_first))
# a simple stemmer that only takes care of the trailing s could for example look like this

def stem(sentence):

    stemmed_sentence = []

    for word in sentence.split():

        stemmed_word = re.findall(r'^(.*?)(s)?$', word)[0][0]

        stemmed_sentence.append(stemmed_word)

    return ' '.join(stemmed_sentence)



# the stemmer works pretty good on this example

example = 'The quick brown fox jumps over the lazy dog.'

print(stem(example))



# a lot of meaning is lost with this stemmer

example_2 = 'He was on the bus with his abs'

print(stem(example_2))
# I can imagine, that for very simple tools there is a simple lookup table

lemma_lookup = {'go': 'go',

               'went': 'go',

               'goes': 'go',

               'gone': 'go',

               'going': 'go',

               'jumps': 'jump',

               'jumped': 'jump',

               'jumping': 'jump'}



def lemma(sentence):

    for word in sentence.lower().split():

        print(lemma_lookup.get(word, word))



example = 'The quick brown fox jumps over the lazy dog.'

lemma(example)
stop_words = ['the', 'over', 'by', 'to', 'from']

example = 'The quick brown fox jumps over the lazy dog.'



#removing stop words is really simple with a list comprehension

[word for word in example.lower().split() if word not in stop_words]
#example of one hot encoding

print('-' * 100)



sentence = 'the quick brown fox jumps over the lazy dog'

row_lookup = {}



print('Origninal sentence: {}'.format(sentence))

print('-' * 100)



# unique words in the corpus represent the number of rows in the matrix

row_names = set(sentence.split())





for i, row in enumerate(row_names):

    row_lookup[row] = i

    



rows = len(row_names)

print('Row Encodings: ', row_lookup)

print('Column Encodings: ', sentence)

# the length of the sentence is the number of columns

columns = len(sentence.split())



print('-' * 100)

one_hot = np.zeros((rows, columns))



for i, column in enumerate(sentence.split()):

    one_hot[row_lookup[column], i] = 1



print(one_hot)



print('-' * 100)
#creating bag of words

from collections import Counter

words = []



sentence_1 = 'the quick brown fox jumps over the lazy dog'

sentence_2 = 'other word'



words_1 = [word for word in sentence_1.split()] 

words_2 = [word for word in sentence_2.split()]

unique_words = set(words_1 + words_2)



counter_1 = Counter(words_1)

counter_2 = Counter(words_2)



bag_1 = {}

bag_2 = {}

for word in unique_words:

    bag_1[word] = counter_1[word]

    bag_2[word] = counter_2[word]



print('-'*120)

print('First bag of words')

print(bag_1)

print('-'*120)

print('Second bag of words')

print(bag_2)

print('-'*120)

# calculate tf idf with pandas

# lets look at the following sentences to understand TF-IDF

sentence_1 = 'the quick brown fox jumps over the lazy dog'

sentence_2 = 'the lazy and dirty dog enjoys his meal'

sentence_3 = 'the fox hunts and eats the chicken'

sentence_4 = 'the angry hunter wants to avenge the chicken'

sentence_5 = 'the hunter can`t find the fox and punishes the dog'

sentence_6 = 'the dog is hungry'

sentence_7 = 'the dog eats the fox'



#will be used for pandas df index

index=['sentence_1', 'sentence_2', 'sentence_3', 'sentence_4', 'sentence_5', 'sentence_6', 'sentence_7']



#for easier looping

corpus=[sentence_1, sentence_2, sentence_3, sentence_4, sentence_5, sentence_6, sentence_7]



#'document' : Counter('token: count_in_document')

counters = {}

for idx, document in enumerate(corpus):

    counters[index[idx]] = Counter(document.split())

      

df = pd.DataFrame(counters).transpose()



row_sum = df.sum(axis=1)

col_count = df.count(axis=0)



df.fillna(value=0, inplace=True)



#term frequency

tf = df.div(row_sum, axis='index')

#inverse document frequency

idf = np.log(len(corpus) / col_count)



#finally tf-idf

tf_idf = tf * idf

tf_idf
# we can use numpy to calculate similarity between the TF-IDF representaion

def similarity(doc_1, doc_2):

    #numerator

    num = np.dot(doc_1, doc_2)

    #denominator

    norm_a = np.linalg.norm(doc_1)

    norm_b = np.linalg.norm(doc_2)

    den = norm_a * norm_b

    return num * den



# sentence 1 and 4 are very dissimilar (no common words)

print(similarity(tf_idf.loc['sentence_1', :], tf_idf.loc['sentence_4', :]))



# sentence 1 and 7 are more similar, because the both have dog and fox mentioned

print(similarity(tf_idf.loc['sentence_1', :], tf_idf.loc['sentence_7', :]))
# We start by working with dummy corpus to learn the libraries

# We introduce some punctuation e.t.c. to test how good the libraries are

corpus = [

    'The quick brown fox jumps over the lazy dog.',

    'The lazy and dirty dog enjoys his meal.',

    'The fox hunts and eats the chicken.',

    'The angry hunter wants to avenge the chicken.',

    'The hunter can`t find the fox and punishes the dog.',

    'The dog is hungry!!!',

    "The dog doesn't let the fox hunt a chicken again.",

    'The hunter went to the doctor!!!'

]



corpus
#spaCy

# Load English tokenizer, tagger, parser, NER and word vectors

nlp = spacy.load("en_core_web_sm")
doc = nlp(corpus[0])



#at first clance this is just a sentence

print(doc)



#But we receive a Doc class

#According to spaCy documentation "A Doc is a sequence of Token objects"

print(type(doc))
#lets loop over the doc object



#https://spacy.io/usage/linguistic-features

    #Text: The original word text.

    #Lemma: The base form of the word.

    #POS: The simple UPOS part-of-speech tag.

    #Tag: The detailed part-of-speech tag.

    #Dep: Syntactic dependency, i.e. the relation between tokens.

    #Shape: The word shape – capitalization, punctuation, digits.

    #is alpha: Is the token an alpha character?

    #is stop: Is the token part of a stop list, i.e. the most common words of the language?



for token in doc:

    print('TOKEN: ', token.text, '\t', '| LEMMA: ', token.lemma_, '   \t', '| POS: ', token.pos_, '\t' \

          '| TAG: ', token.tag_, '\t',  '| DEPENDENCY: ', token.dep_, '     \t', \

          '| SHAPE: ',  token.shape_, '   \t', 'ISALPHA: ', token.is_alpha, '\t', 'ISSTOPWORD: ', token.is_stop)
# spacy has an explain method to help with the language terminology

spacy.explain('ADJ')
# lets take it one step at a time and look at a simple sentence

print(corpus[-1:])



doc = nlp(corpus[-1])



print('\nLEMMAS:')

print('-'*30)

# How do the lemmas look like?

# The tokens look natural. And it correctly transforms 'went' into 'go'

for token in doc:

    print(token.lemma_, end=" ")

    

# What are the stop words

print('\n\nSTOP WORDS:')

print('-'*30)

for token in doc:

    if token.is_stop: print(token.text, end=" ")
# an additional advantage of spacy is that it allows you to draw the relationships in a sentence

displacy.render(doc, style="dep")
# lets reduce the pipeline

# we don't need all component of spacy, removing parts in the pipeline improves the performance

# https://spacy.io/usage/processing-pipelines#pipelines

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])



#nlp = spacy.load("en_core_web_sm")
# inspiration for the function

# https://towardsdatascience.com/turbo-charge-your-spacy-nlp-pipeline-551435b664ad

def lemmatize(text):

    doc = nlp(text)

    #remove stop words and punctuation and return the lemmas ow words

    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.pos_ == 'PUNCT'])



for df in concat:

    df['lemma'] = df['text'].apply(lemmatize)
# looks reasonable enough

train_df.loc[:, ['text', 'lemma']].head(10)
# we use the TF-IDF Class to generate the vectorized form of the corpus

vectorizer = TfidfVectorizer()

corpus_train = train_df.loc[:, 'lemma']

corpus_test = test_df.loc[:, 'lemma']



# I don't quite understand how the fit function would work for the test data if we fit the data on the training set

# The TF has to be calculated for each document, but the IDF has to be calculated based on each word and the documents that the word appears

# do we use the test data, the test + train data or only the train data to calculate the idf (I assume train data)

# if we fit the model on the training data I would assume, that only the training data is used, but how is dealt with new words



#building a pipeline in sklearn might be a good idea overall

X_train = vectorizer.fit_transform(corpus_train)

X_test = vectorizer.transform(corpus_test)
# there are 7613 sentences and 12394 words

print(X_train.shape)

print(X_test.shape)
# here we can get all the available words

#print(vectorizer.get_feature_names())
y_train = train_df['target']
# Logistic Regression could be used as a baseline

log_reg = LogisticRegression()

scores = cross_val_score(log_reg, X_train, y_train, cv=10, scoring='f1', n_jobs=-1)
print(scores)

print(scores.mean())
# fit the model on all data points for submission

log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)
f1_score(y_train, log_reg.predict(X_train))
# Logistic Regression could be used as a baseline

knn = KNeighborsClassifier(n_neighbors=100)

scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='f1', n_jobs=-1)
print(scores)

print(scores.mean())
rf = RandomForestClassifier(n_estimators=100)

#this part takes a lot of time if you use cv

# scores = cross_val_score(rf, X_train, y_train, cv=10, scoring='f1', n_jobs=-1)

rf.fit(X_train, y_train)
f1_score(y_train, rf.predict(X_train))
# SOME PARAMETERS OF XGBOOST

# eta = learning_rate (default=0.3)

# gamma = min_split_loss (default=0)

# objective = loss_function (default=reg:squarederror), we will use binary:logistic



#here we create a cross val cross validation set in order to be able to use early stopping

X_train_small, X_cv, y_train_small, y_cv = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)





#we determined at the top if we are using the gpu

#if device == 'GPU':

    # better turn on the gpu

    #xgb_clf = xgb.XGBClassifier(n_estimators=1000, gamma=0.1, objective='binary:logistic', tree_method='gpu_hist')



    #xgb_clf.fit(X_train_small, y_train_small, eval_set=[(X_cv, y_cv)], eval_metric="logloss", verbose=True, early_stopping_rounds=10)



    # in case we want to use stratified cross validation

    #scores = cross_val_score(xgb_clf, X_train, y_train, cv=10, scoring='f1')
#print(scores)

#print(scores.mean())
#if device == 'GPU':

#    #xgb_clf.fit(X_train, y_train)

#    f1_score(y_train, xgb_clf.predict(X_train))
tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_df.text)
# over 15000 unique tokens

# that is probably too much for such a small dataset size

print(len(tokenizer.word_index))
X_train = tokenizer.texts_to_matrix(train_df.text, mode='binary')

X_test = tokenizer.texts_to_matrix(test_df.text, mode='binary')
X_train.shape
# keras model

#mlp_model = keras.models.Sequential()

#mlp_model.add(layers.Input(shape=(X_train.shape[1],)))

#mlp_model.add(layers.Dense(100, activation='relu'))

#mlp_model.add(layers.Dense(20, activation='relu'))

#mlp_model.add(layers.Dense(1, activation='sigmoid'))
#callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
#mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#history = mlp_model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0.25, callbacks=[callback])
# keras model

'''

embed_model = keras.models.Sequential()

#embed_model.add(layers.Input(shape=(X_train.shape[1],)))

embed_model.add(layers.Embedding(X_train.shape[1], 100, input_length=X_train.shape[1]))

embed_model.add(layers.Flatten())

embed_model.add(layers.Dense(100, activation='relu'))

embed_model.add(layers.Dense(10, activation='relu'))

embed_model.add(layers.Dense(1, activation='sigmoid'))

'''
#callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
#embed_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['acc'])
#history = embed_model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0.25, callbacks=[callback])
'''

rnn_model = keras.models.Sequential()

rnn_model.add(layers.Input(shape=(None, X_train.shape[1])))

#rnn_model.add(layers.Embedding(X_train.shape[1], 10))

rnn_model.add(layers.SimpleRNN(10))

rnn_model.add(layers.Dense(10, activation='relu'))

rnn_model.add(layers.Dense(1, activation='sigmoid'))

'''
#rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#history = rnn_model.fit(X_train.reshape((7613, 1, 15365)), y_train, epochs=200, batch_size=128, validation_split=0.25, callbacks=[callback])
max_words = 5000

embedding_dim = 100

maxlen = 100

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(train_df.text)
X_train = tokenizer.texts_to_sequences(train_df.text)

X_test = tokenizer.texts_to_sequences(test_df.text)
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)

X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)
word_index = tokenizer.word_index
X_train[0]
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
'''

lstm_model = keras.models.Sequential()

lstm_model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))

lstm_model.add(layers.LSTM(32,return_sequences=True))

lstm_model.add(layers.LSTM(32,return_sequences=True))

lstm_model.add(layers.LSTM(32,return_sequences=True))

lstm_model.add(layers.LSTM(32))

lstm_model.add(layers.Dense(1, activation='sigmoid'))

'''
#lstm_model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['acc'])
#history = lstm_model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.25, callbacks=[callback])
'''

cnn_model = keras.models.Sequential()

cnn_model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))

cnn_model.add(layers.Conv1D(16, 7, activation='relu'))

cnn_model.add(layers.Dropout(0.5))

cnn_model.add(layers.MaxPooling1D(5))

cnn_model.add(layers.Conv1D(32, 7, activation='relu'))

cnn_model.add(layers.Dropout(0.5))

cnn_model.add(layers.GlobalMaxPooling1D())

cnn_model.add(layers.Dense(1, activation='sigmoid'))

'''
#cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#history = cnn_model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0.25, callbacks=[callback])
import tensorflow_hub as hub

import tensorflow as tfhub

import tensorflow_datasets as tfds

import tensorflow as tf

hub_model = keras.Sequential([

    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1", dtype=tf.string, input_shape=[], output_shape=[50]),

    layers.Dense(128, activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(10, activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')

])
hub_model.summary()
hub_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train = train_df['text'].to_numpy()

y_train = train_df['target'].to_numpy()
X_test = test_df['text'].to_numpy()
print(X_train.shape)

print(y_train.shape)
X_train_tr = X_train[:6000]

X_train_cv = X_train[6000:] 



y_train_tr = y_train[:6000]

y_train_cv = y_train[6000:]
train_data = tf.data.Dataset.from_tensor_slices((X_train_tr, y_train_tr))

validation_data = tf.data.Dataset.from_tensor_slices((X_train_cv, y_train_cv))
train_data
#train_examples_batch, train_labels_batch = next(iter(train_dataset.batch(10)))
#train_examples_batch
history = hub_model.fit(train_data.shuffle(10000).batch(512),

                    epochs=20,

                    verbose=1,

                    validation_data=validation_data.batch(512))
model = hub_model



# for sklearn

#y_pred = model.predict(X_test)



# for keras

y_pred = model.predict_classes(X_test)

y_pred = y_pred.reshape(y_pred.shape[0],)



submit_df = pd.DataFrame({'id': test_df.index, 'target': y_pred})
submit_df.shape
submit_df.head()
submit_df.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")