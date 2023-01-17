# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import pandas as pd

import string

import re

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter

import nltk

import spacy

import sys

from spacy.lang.en import English

import en_core_web_sm

from nltk.corpus import wordnet as wn

from nltk.stem.wordnet import WordNetLemmatizer



print(tf.__version__)  # 2.0.0-beta0
# Run this code for the first time, to install the libraries and download wordnet

# %reset

# !{sys.executable} -m pip install spacy

# !{sys.executable} -m spacy download en

# !{sys.executable} -m pip install pyLDAvis

# !{sys.executable} -m pip install gensim

# nltk.download('stopwords')

# nltk.download('wordnet')
df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)

df = df[['headline', 'is_sarcastic']]

df.head()
# check for columns with null values

df.is_sarcastic.isnull().any() # no missing values in is_sarcastic column

df.headline.isnull().any() # no missing values in headline column
df['headline'] = df.headline.apply(lambda x:x.lower())  # convert all words in headline into lower case 

df['headline'] = df.headline.apply(lambda x: ' '.join(word.strip(string.punctuation) for word in x.split()))  # remove all punctuations in headline
df['headline_count'] = df.headline.apply(lambda x: len(list(x.split())))

df['headline_unique_word_count'] = df.headline.apply(lambda x: len(set(x.split())))

df['headline_has_digits'] = df.headline.apply(lambda x: bool(re.search(r'\d', x)))

df
sarcastic_dat = df.groupby('is_sarcastic').count()

sarcastic_dat.index = ['Non-sarcastic','Sarcastic']

plt.xlabel('Type of headlines (Sarcastic & Non-sarcastic)')

plt.ylabel('Frequencies of headlines')

plt.xticks(fontsize=10)

plt.title('Frequencies of Sarcastic vs Non-sarcastic headlines')

bar_graph = plt.bar(sarcastic_dat.index, sarcastic_dat.headline_count)

bar_graph[1].set_color('r')

plt.show()





plt.xlabel('Type of headlines (Sarcastic & Non-sarcastic)')

plt.ylabel('Proportion of headlines')

plt.xticks(fontsize=10)

plt.title('Proportion of Sarcastic vs Non-sarcastic headlines')

bar_graph = plt.bar(sarcastic_dat.index, sarcastic_dat.headline_count / sarcastic_dat.headline_count.sum())

bar_graph[1].set_color('r')

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.show()



# This is not an imbalanced class dataset

# Non-sarcastic    0.56

# Sarcastic        0.44

round(sarcastic_dat.headline_count / sarcastic_dat.headline_count.sum(), 2)
all_dat = df.groupby('headline_count').count()

sarcastic_dat1 = df[df.is_sarcastic==1]

sarcastic_dat = sarcastic_dat1.groupby('headline_count').count()

not_sarcastic_dat1 = df[df.is_sarcastic==0]

not_sarcastic_dat = not_sarcastic_dat1.groupby('headline_count').count()



plt.xlabel('Different lengths of headline')

plt.ylabel('Frequencies of headline length')

plt.xticks(fontsize=10)

plt.title('Distribution of headline length for entire dataset')

bar_graph = plt.bar(all_dat.index, all_dat.headline)

bar_graph[8].set_color('r')

plt.axvline(df.headline_count.mean(), color='k', linestyle='dashed', linewidth=1)  # median is 10 words in a headline

plt.show()



plt.xlabel('Different lengths of sarcastic headline')

plt.ylabel('Frequencies of sarcastic headline length')

plt.xticks(fontsize=10)

plt.title('Distribution of headline length for sarcastic dataset')

bar_graph = plt.bar(sarcastic_dat.index, sarcastic_dat.headline)

bar_graph[7].set_color('r')

plt.axvline(sarcastic_dat1.headline_count.mean(), color='k', linestyle='dashed', linewidth=1)  # median is 10 words in a headline

plt.show()





plt.xlabel('Different lengths of non-sarcastic headline')

plt.ylabel('Frequencies of non-sarcastic headline length')

plt.xticks(fontsize=10)

plt.title('Distribution of headline length for non-sarcastic dataset')

bar_graph = plt.bar(not_sarcastic_dat.index, not_sarcastic_dat.headline)

bar_graph[8].set_color('r')

plt.axvline(not_sarcastic_dat1.headline_count.mean(), color='k', linestyle='dashed', linewidth=1)  # median is 10 words in a headline

plt.show()



# difference in the length of sarcastic and non-sarcastic headlines is not significant. 

# median and mean length of headlines is around 10 words
digits_dat = df.groupby('headline_has_digits').count()

digits_dat.index = ['Has Numbers in Headline','Does not have Numbers in Headline']





plt.xlabel('Type of headlines')

plt.ylabel('Frequencies of headlines')

plt.xticks(fontsize=10)

plt.title('Frequencies of headlines with Numbers vs No numbers')

bar_graph = plt.bar(digits_dat.index, digits_dat.headline / digits_dat.headline_count.sum())

bar_graph[1].set_color('r')

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.show()





sarcastic_digits_dat = df[df.is_sarcastic==1].groupby('headline_has_digits').count()

sarcastic_digits_dat.index = ['Has Numbers in Headline','Does not have Numbers in Headline']





plt.xlabel('Type of headlines')

plt.ylabel('Frequencies of headlines')

plt.xticks(fontsize=10)

plt.title('Frequencies of Sarcastic headlines with Numbers vs No numbers')

bar_graph = plt.bar(sarcastic_digits_dat.index, sarcastic_digits_dat.headline / sarcastic_digits_dat.headline_count.sum())

bar_graph[1].set_color('r')

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.show()





not_sarcastic_digits_dat = df[df.is_sarcastic==0].groupby('headline_has_digits').count()

not_sarcastic_digits_dat.index = ['Has Numbers in Headline','Does not have Numbers in Headline']





plt.xlabel('Type of headlines')

plt.ylabel('Frequencies of headlines')

plt.xticks(fontsize=10)

plt.title('Frequencies of Non-sarcastic headlines with Numbers vs No numbers')

bar_graph = plt.bar(not_sarcastic_digits_dat.index, not_sarcastic_digits_dat.headline / not_sarcastic_digits_dat.headline_count.sum())

bar_graph[1].set_color('r')

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.show()



print(round(digits_dat.headline / digits_dat.headline_count.sum(),2))

print(round(sarcastic_digits_dat.headline / sarcastic_digits_dat.headline_count.sum(),2))

print(round(not_sarcastic_digits_dat.headline / not_sarcastic_digits_dat.headline_count.sum(),2))



# difference in the use of numbers/statistics in sarcastic and non-sarcastic headlines is not significant. 

# ~85% headlines uses numbers
nlp = en_core_web_sm.load()

parser = English()

en_stop = set(nltk.corpus.stopwords.words('english'))





def tokenize(text):

    """this function is to tokenize the headline into a list of individual words"""

    lda_tokens = []

    tokens = parser(text)  # need to use parser for python to treat the list as words

    for token in tokens:

        if token.orth_.isspace():  # to ignore any whitespaces in the headline, so that token list does not contain whitespaces 

            continue

        elif token.like_url:

            lda_tokens.append('URL')

        elif token.orth_.startswith('@'):

            lda_tokens.append('SCREEN_NAME')

        else:

            lda_tokens.append(token.lower_)   # tokens (headlines) are already in lowercase

    return lda_tokens





def get_lemma(word):

    """this function is to lemmatize the words in a headline into its root form"""

    lemma = wn.morphy(word)  # converts the word into root form from wordnet

    if lemma is None:

        return word

    else:

        return lemma

    



def prepare_text_for_lda(text):

    tokens = tokenize(text)  # parse and tokenize the headline into a list of words

    tokens = [token for token in tokens if len(token) > 4]  # remove headlines with only length of 4 words or less

    tokens = [token for token in tokens if token not in en_stop]  # remove stopwords in the headline

    tokens = [get_lemma(token) for token in tokens]  # lemmatize the words in the headline

    return tokens
text_data = []

for headline in df.headline:

    tokens = prepare_text_for_lda(headline)

    text_data.append(tokens)
from gensim import corpora

import pickle



dictionary = corpora.Dictionary(text_data)  # Convert all headlines into a corpus of words, with each word as a token

corpus = [dictionary.doc2bow(text) for text in text_data]  # Convert each headline (a list of words) into the bag-of-words format. (Word ID, Count of word)

pickle.dump(corpus, open('corpus.pkl', 'wb'))  

dictionary.save('dictionary.gensim')  # takes a while to run the dictionary and corpus
import gensim



NUM_TOPICS = [3, 5, 10]

# passes: Number of passes through the corpus during training

# alpha: priori on the distribution of the topics in each document.

# The higher the alpha, the higher the likelihood that document contains a wide range of topics, vice versa. 

# beta: priori on the distribution of the words in each topic.

# The higher the beta, the higher the likelihood that topic contains a wide range of words, vice versa.

# we do not alter / fine tune the default values of alpha and beta

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS[1], id2word=dictionary, passes=15)

ldamodel.save('model5.gensim')

topics = ldamodel.print_topics(num_words=5)

topics
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 3, id2word=dictionary, passes=15)

ldamodel.save('model3.gensim')

topics = ldamodel.print_topics(num_words=5)

topics
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, id2word=dictionary, passes=15)

ldamodel.save('model10.gensim')

topics = ldamodel.print_topics(num_words=5)

topics
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')

corpus = pickle.load(open('corpus.pkl', 'rb'))

lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

import pyLDAvis.gensim

lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

pyLDAvis.display(lda_display)
lda3 = gensim.models.ldamodel.LdaModel.load('model3.gensim')

lda_display3 = pyLDAvis.gensim.prepare(lda3, corpus, dictionary, sort_topics=False)

pyLDAvis.display(lda_display3)
lda10 = gensim.models.ldamodel.LdaModel.load('model10.gensim')

lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=False)

pyLDAvis.display(lda_display10)
from numpy import mean



sarcastic = list(df.is_sarcastic == 1)

tuple_list = []

for headline in sarcastic:

    sarcastic = lda10[corpus[headline]]

    for tuple_ in sarcastic:

        tuple_list.append(tuple_)



print('For LDA model with 10 clusters:')

print('\nFor Sarcastic Dataset:')

print([(uk, mean([vv for kk,vv in tuple_list if kk==uk])) for uk in set([k for k,v in tuple_list])])



not_sarcastic = list(df.is_sarcastic == 0)

tuple_list = []

for headline in not_sarcastic:

    not_sarcastic = lda10[corpus[headline]]

    for tuple_ in not_sarcastic:

        tuple_list.append(tuple_)

        



print('\nFor Non-sarcastic Dataset:')

print([(uk, mean([vv for kk,vv in tuple_list if kk==uk])) for uk in set([k for k,v in tuple_list])])



# LDA model with 10 clusters not differentiable between sarcastic and not sarcastic headlines.

# Not very interpretable
sarcastic = list(df.is_sarcastic == 1)

tuple_list = []

for headline in sarcastic:

    sarcastic = lda[corpus[headline]]

    for tuple_ in sarcastic:

        tuple_list.append(tuple_)



print('For LDA model with 5 clusters:')

print('For Sarcastic Dataset:')

print([(uk, mean([vv for kk,vv in tuple_list if kk==uk])) for uk in set([k for k,v in tuple_list])])



not_sarcastic = list(df.is_sarcastic == 0)

tuple_list = []

for headline in not_sarcastic:

    not_sarcastic = lda[corpus[headline]]

    for tuple_ in not_sarcastic:

        tuple_list.append(tuple_)

        



print('\nFor Non-sarcastic Dataset:')

print([(uk, mean([vv for kk,vv in tuple_list if kk==uk])) for uk in set([k for k,v in tuple_list])])



# LDA model with 5 clusters not differentiable between sarcastic and not sarcastic headlines.

# Not very interpretable
sarcastic = list(df.is_sarcastic == 1)

tuple_list = []

for headline in sarcastic:

    sarcastic = lda3[corpus[headline]]

    for tuple_ in sarcastic:

        tuple_list.append(tuple_)



print('For LDA model with 3 clusters:')

print('For Sarcastic Dataset:')

print([(uk, mean([vv for kk,vv in tuple_list if kk==uk])) for uk in set([k for k,v in tuple_list])])



not_sarcastic = list(df.is_sarcastic == 0)

tuple_list = []

for headline in not_sarcastic:

    not_sarcastic = lda3[corpus[headline]]

    for tuple_ in not_sarcastic:

        tuple_list.append(tuple_)

        



print('\nFor Non-sarcastic Dataset:')

print([(uk, mean([vv for kk,vv in tuple_list if kk==uk])) for uk in set([k for k,v in tuple_list])])



# LDA model with 3 clusters not differentiable between sarcastic and not sarcastic headlines.

# Not very interpretable
train_data, test_data = train_test_split(df[['headline', 'is_sarcastic']], test_size=0.1)  # randomly splitting 10% of dataset to be training dataset 



training_sentences = list(train_data['headline'])

training_labels = list(train_data['is_sarcastic'])



testing_sentences = list(test_data['headline'])

testing_labels = list(test_data['is_sarcastic'])

training_labels_final = np.array(training_labels)

testing_labels_final = np.array(testing_labels)
vocab_size = 10000   # limit vector of words to the top 10,000 words

embedding_dim = 16

max_length = 120

trunc_type='post'

oov_tok = "<OOV>"





from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)



testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

testing_padded = pad_sequences(testing_sequences,maxlen=max_length)



# no lemmatization, removal of stop words and stemming of headlines as we would like to maintain the syntax, literature integrity, sequence of words in LSTM.
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])



def decode_review(text):

    return ' '.join([reverse_word_index.get(i, '?') for i in text])
# Model Definition with BiRNN (GRU)

# with L1 Lasso Regularization, for feature selection

# Dropout, for robustness of recurrent neural networks

# Batch Normalization, to stabilize and perhaps accelerate the learning process



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),

    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 10

history = model.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))
import matplotlib.pyplot as plt





def plot_graphs(history, string):

    plt.plot(history.history[string])

    plt.plot(history.history['val_'+string])

    plt.xlabel("Epochs")

    plt.ylabel(string)

    plt.legend([string, 'val_'+string])

    plt.show()



plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

plt.show()
# Model Definition with BiRNN (GRU)

# with L2 Ridge Regularization

# Dropout, for robustness of recurrent neural networks

# Batch Normalization, to stabilize and perhaps accelerate the learning process



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),

    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.003), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.003), activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 10

history = model.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))
plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

plt.show()
# Model Definition with BiRNN (LSTM)

# with L1 Lasso Regularization, for feature selection

# Dropout, for robustness of recurrent neural networks

# Batch Normalization, to stabilize and perhaps accelerate the learning process



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='sigmoid')

])



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 10

history = model.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))
plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

plt.show()
# Model Definition with BiRNN (LSTM)

# with L2 Ridge Regularization

# Dropout, for robustness of recurrent neural networks

# Batch Normalization, to stabilize and perhaps accelerate the learning process



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.003), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.003), activation='sigmoid')

])



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 10

history = model.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))
plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

plt.show()
# Model Definition with CNN (Conv1D)

# with L1 Lasso Regularization, for feature selection

# Dropout, for robustness

# Batch Normalization, to stabilize and perhaps accelerate the learning process



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Conv1D(128, 5, activation='relu'),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 10

history = model.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))
plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

plt.show()
# Model Definition with CNN (Conv1D)

# with L2 Ridge Regularization

# Dropout, for robustness

# Batch Normalization, to stabilize and perhaps accelerate the learning process



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Conv1D(128, 5, activation='relu'),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.003), activation='relu'),

    # tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.003), activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

num_epochs = 10

history = model.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))
plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

plt.show()
# Model Definition with CNN (Conv1D)

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Conv1D(128, 1, activation='relu'),

    tf.keras.layers.MaxPooling1D(2, padding="same"),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l1(0.005), activation='relu'),

    # tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.005), activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

plt.show()