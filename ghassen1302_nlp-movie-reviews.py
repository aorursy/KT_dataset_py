!pip install num2words
import tensorflow as tf
from keras import layers
import keras
import matplotlib.pyplot as plt

import nltk, re, string
import os
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer       
from num2words import num2words

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import random
from collections import Counter
import tensorflow_hub as hub
from tensorflow.keras import regularizers
from keras.callbacks import ModelCheckpoint

import numpy as np
import os
import pickle
from string import punctuation
import requests

plt.style.use('fivethirtyeight') 
%matplotlib inline
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
import os
import pandas as pd

# 1 --> pos
# 0 --> neg

data = pd.DataFrame()
comment = []
label = []
for filename in os.listdir("aclImdb/train/pos"):
    with open("aclImdb/train/pos/"+str(filename), 'r', encoding="utf-8") as f: # open in readonly mode 
        comment.append(f.readlines())
        label.append(1)

for filename in os.listdir("aclImdb/train/neg"):
    with open("aclImdb/train/neg/"+str(filename), 'r', encoding="utf-8") as f: # open in readonly mode 
        comment.append(f.readlines())
        label.append(0)

for filename in os.listdir("aclImdb/test/pos"):
    with open("aclImdb/test/pos/"+str(filename), 'r', encoding="utf-8") as f: # open in readonly mode 
        comment.append(f.readlines())
        label.append(1)

for filename in os.listdir("aclImdb/test/neg"):
    with open("aclImdb/test/neg/"+str(filename), 'r', encoding="utf-8") as f: # open in readonly mode 
        comment.append(f.readlines())
        label.append(0)
        
data['comment'] = comment
data['y'] = label
X = pd.DataFrame(list(data['comment']))
Y = pd.DataFrame(list(data['y']))
"""Module to explore data.
Contains functions to help study, visualize and understand datasets.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


def get_num_classes(labels):
    """Gets the total number of classes.
    # Arguments
        labels: list, label values.
            There should be at lease one sample for values in the
            range (0, num_classes -1)
    # Returns
        int, total number of classes.
    # Raises
        ValueError: if any label value in the range(0, num_classes - 1)
            is missing or if number of classes is <= 1.
    """
    num_classes = max(labels) + 1
    missing_classes = [i for i in range(num_classes) if i not in labels]
    if len(missing_classes):
        raise ValueError('Missing samples with label value(s) '
                         '{missing_classes}. Please make sure you have '
                         'at least one sample for every label value '
                         'in the range(0, {max_class})'.format(
                            missing_classes=missing_classes,
                            max_class=num_classes - 1))

    if num_classes <= 1:
        raise ValueError('Invalid number of labels: {num_classes}.'
                         'Please make sure there are at least two classes '
                         'of samples'.format(num_classes=num_classes))
    return num_classes


def get_num_words_per_sample(sample_texts):
    """Gets the median number of words per sample given corpus.
    # Arguments
        sample_texts: list, sample texts.
    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)


def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range=(1, 2),
                                          num_ngrams=50):
    """Plots the frequency distribution of n-grams.
    # Arguments
        samples_texts: list, sample texts.
        ngram_range: tuple (min, mplt), The range of n-gram values to consider.
            Min and mplt are the lower and upper bound values for the range.
        num_ngrams: int, number of n-grams to plot.
            Top `num_ngrams` frequent n-grams will be plotted.
    """
    # Create args required for vectorizing.
    kwargs = {
            'ngram_range': (1, 1),
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    # This creates a vocabulary (dict, where keys are n-grams and values are
    # idxices). This also converts every text to an array the length of
    # vocabulary, where every element idxicates the count of the n-gram
    # corresponding at that idxex in vocabulary.
    vectorized_texts = vectorizer.fit_transform(sample_texts)

    # This is the list of all n-grams in the index order from the vocabulary.
    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(num_ngrams, len(all_ngrams))
    # ngrams = all_ngrams[:num_ngrams]

    # Add up the counts per n-gram ie. column-wise
    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])
    ngrams = list(all_ngrams)[:num_ngrams]
    counts = list(all_counts)[:num_ngrams]

    idx = np.arange(num_ngrams)
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('Frequency distribution of n-grams')
    plt.xticks(idx, ngrams, rotation=45)
    plt.show()


def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.
    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()


def plot_class_distribution(labels):
    """Plots the class distribution.
    # Arguments
        labels: list, label values.
            There should be at lease one sample for values in the
            range (0, num_classes -1)
    """
    num_classes = get_num_classes(labels)
    count_map = Counter(labels)
    counts = [count_map[i] for i in range(num_classes)]
    idx = np.arange(num_classes)
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Class distribution')
    plt.xticks(idx, idx)
    plt.show()
def remove_html(comment):
    return comment.replace('<br />', ' ')

def lower_comment(comment):
    return comment.lower()

def remove_ponctuation(comment):
    for punc in string.punctuation:
        comment = comment.replace(punc, '')
    return comment

def tokenize_comment(comment):
    return comment.split()

def remove_stop_words(comment_tokenized):
    stopwords_english = stopwords.words('english')
    comment_tokenized_cleaned = []
    for word in comment_tokenized:
        if(word not in stopwords_english):
            comment_tokenized_cleaned.append(word)
    return comment_tokenized_cleaned

def convert_numbers_to_words(comment_tokenized):
    comment_tokenized_cleaned = []
    for word in comment_tokenized:  # Go through every word in your tokens list
        try:
            comment_tokenized_cleaned.append(num2words(word))
        except:
            comment_tokenized_cleaned.append(word)
    return comment_tokenized_cleaned

def remove_ponctuation_from_tokenized(comment_tokenized):
    comment_tokenized_cleaned = []
    for word in comment_tokenized:
        comment_tokenized_cleaned.append(remove_ponctuation(word))
    
    return comment_tokenized_cleaned

def stem_words(comment_tokenized):
    stemmer = PorterStemmer()
    comment_tokenized_cleaned = []
    for word in comment_tokenized:  # Go through every word in your tokens list
        comment_tokenized_cleaned.append(stemmer.stem(word))   # stemming word
    return comment_tokenized_cleaned


# Clean the data.
cleaned_comments = []
for i in tqdm(range (len(list(X[0])))):
    text = remove_html(list(X[0])[i])
    text = lower_comment(text)
    text = remove_ponctuation(text)
    text_tokenized = tokenize_comment(text)
    text_tokenized = convert_numbers_to_words(text_tokenized)
    text_tokenized = remove_stop_words(text_tokenized)
    text_tokenized = remove_ponctuation_from_tokenized(text_tokenized)
    text_tokenized = stem_words(text_tokenized)
    
    cleaned_comments.append(text_tokenized)
def get_num_words_per_sample(sample_texts):
    """Gets the median number of words per sample given corpus.
    # Arguments
        sample_texts: list, sample texts.
    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s) for s in sample_texts]
    return np.median(num_words)
cleaned_comments_sentance = []
for sentance in cleaned_comments:
    ch = ""
    for word in sentance:
        ch = ch + word + " "
    cleaned_comments_sentance.append(ch)
get_num_classes(list(Y[0]))  # Number of classes.
get_num_words_per_sample(cleaned_comments)   # Median number of words per sample.
vocabulary_size = 20000
text_length = 90
plot_frequency_distribution_of_ngrams(cleaned_comments_sentance, num_ngrams=15)
plot_sample_length_distribution(cleaned_comments_sentance)
plot_class_distribution(list(Y[0]))
# 561 < 1500
len(list(X[0])) / get_num_words_per_sample(cleaned_comments)
len(cleaned_comments_sentance)
stopwords_english = stopwords.words('english')
tfidf = TfidfVectorizer(ngram_range=(2, 2), max_features=10000, stop_words=stopwords_english)   # bigram, a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
cleaned_comments_sentance[0]
response = tfidf.fit_transform(cleaned_comments_sentance)
response.shape[0]
len(tfidf.get_feature_names())
df = pd.DataFrame(response.toarray(), columns=tfidf.get_feature_names())
df
# # Create an SelectKBest object to select features with the best 5000 ANOVA F-Values
# fvalue_selector = feature_selection.SelectKBest(feature_selection.f_classif, k=5000)   # k: Number of top features to select. The “all” option bypasses selection, for use in a parameter search.

# # Apply the SelectKBest object to the features and target
# response_best = fvalue_selector.fit_transform(response.toarray(), np.array(Y))

# cols = fvalue_selector.get_support(indices=True)

# choosen_columns = []
# for i in cols:
#   choosen_columns.append(tfidf.get_feature_names()[i])

# df = pd.DataFrame(response_best, columns=choosen_columns)
# df
# Shuffle the data
# Fisher–Yates shuffle
for i in tqdm(range(len(df)-2)):
  # Shuffle X
  j = random.randint(i, len(df)-1)
  aux = df.iloc[j]
  df.loc[j] = df.iloc[i]
  df.loc[i] = aux
  df.reset_index(drop=True, inplace=True)

  # Shuffle Y
  aux_y = Y.iloc[j]
  Y.loc[j] = Y.iloc[i]
  Y.loc[i] = aux_y
  Y.reset_index(drop=True, inplace=True)

X_train = df[:40000]
X_test = df[40000:50000]
Y_train = Y[:40000]
Y_test = Y[40000:50000]
def get_optimizer(batch_size_var, X_train):
    # Many models train better if you gradually reduce the learning rate during training. 
    # Use optimizers.schedules to reduce the learning rate over time
    N_TRAIN = X_train.shape[0]
    STEPS_PER_EPOCH = N_TRAIN//batch_size_var

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, 
                                                                 decay_steps=STEPS_PER_EPOCH*1000, 
                                                                 decay_rate=1, 
                                                                 staircase=False)
    
    return tf.keras.optimizers.Adam(lr_schedule)

def compile_model(model, loss_func, batch_size_var, epochs_var, X_train, Y_train, X_test, Y_test):
  model.compile(optimizer=get_optimizer(batch_size_var, X_train), 
                loss=loss_func,
                metrics=['accuracy'])

  history = model.fit(
      X_train,
      Y_train,
      batch_size=batch_size_var,
      epochs=epochs_var,
      # We pass some validation for
      # monitoring validation loss and metrics
      # at the end of each epoch
      validation_data=(X_test, Y_test),
  )


  history_dict = history.history
  history_dict.keys()

  acc = history_dict['accuracy']
  val_acc = history_dict['val_accuracy']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']

  epochs = range(1, len(acc) + 1)

  # "bo" is for "blue dot"
  plt.plot(epochs, loss, 'bo', label='Training loss')
  # b is for "solid blue line"
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.show()

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')

  plt.show()
  return model
input_layer = keras.Input(shape=(X_train.shape[1]))  
x = layers.Dense(64, activation="relu")(input_layer)
x = layers.Dense(64, activation="relu")(x)
output_layer = layers.Dense(2, activation='softmax')(x)

model = keras.Model(inputs=input_layer, outputs=output_layer, name="model")
model.summary()
keras.utils.plot_model(model, "model.png", show_shapes=True)
model = compile_model(model, 'sparse_categorical_crossentropy', 64, 10, X_train, Y_train, X_test, Y_test)
test = ["This movie is really bad", "I like this movie", "I didn't enjoy watching this movie", "I did enjoy watching this movie", "It's a waste of time", "I enjoyed watching this movie"]
# Clean the data.
cleaned_comments_test = []
for i in tqdm(range (len(test))):
    text = remove_html(test[i])
    text = lower_comment(text)
    text = remove_ponctuation(text)
    text_tokenized = tokenize_comment(text)
    text_tokenized = convert_numbers_to_words(text_tokenized)
    text_tokenized = remove_stop_words(text_tokenized)
    text_tokenized = remove_ponctuation_from_tokenized(text_tokenized)
    text_tokenized = stem_words(text_tokenized)
    
    cleaned_comments_test.append(text_tokenized)
cleaned_comments_sentance_test = []
for sentance in cleaned_comments_test:
    ch = ""
    for word in sentance:
        ch = ch + word + " "
    cleaned_comments_sentance_test.append(ch)
cleaned_comments_sentance_test
response_test = tfidf.transform(cleaned_comments_sentance_test) 
df_test = pd.DataFrame(response_test.toarray(), columns=tfidf.get_feature_names())
df_test
pred_list = np.argmax(model.predict(df_test), axis=1)   # These are the probabilities of each class (pos or neg)
for i in range(len(pred_list)):
  if(pred_list[i]==0):
    print(test[i]+" --> neg")
  else:
    print(test[i]+" --> pos")

# Testing ResNet architecture

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
input_layer = keras.Input(shape=(), name="Input", dtype=tf.string)  

hub_layer = hub.KerasLayer(embedding, trainable=True, name='embedding')(input_layer)

# Adding dimensions to use SeparableConv2D
x = tf.expand_dims(hub_layer, -1)

x = layers.SeparableConv1D(32, 3, activation="relu", padding="same")(x)
x = layers.SeparableConv1D(64, 3, activation="relu", padding="same")(x)

block_1_output = layers.MaxPooling1D(3, padding="same")(x)

x = layers.SeparableConv1D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.SeparableConv1D(64, 3, activation="relu", padding="same")(x)

block_2_output = layers.add([x, block_1_output])

x = layers.SeparableConv1D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.SeparableConv1D(64, 3, activation="relu", padding="same")(x)

block_3_output = layers.add([x, block_2_output])

x = layers.SeparableConv1D(64, 3, activation="relu", padding="same")(block_3_output)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)

output_layer = tf.keras.layers.Dense(2, activation='softmax')(x)

# Remove extra dimensions
# output_layer = tf.squeeze(output_layer, [0, 1])

second_model = keras.Model(inputs=input_layer, outputs=output_layer, name="model")

second_model.summary()
keras.utils.plot_model(second_model, "model.png", show_shapes=True)
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
input_layer = keras.Input(shape=(), name="Input", dtype=tf.string)  

hub_layer = hub.KerasLayer(embedding, trainable=True, name='embedding')(input_layer)

x = tf.expand_dims(hub_layer, 1)

x = layers.Bidirectional(layers.LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2))(x)

output_layer = tf.keras.layers.Dense(2, activation='softmax')(x)

third_model = keras.Model(inputs=input_layer, outputs=output_layer, name="model")

third_model.summary()
keras.utils.plot_model(third_model, "model.png", show_shapes=True)
from sklearn.model_selection import train_test_split    

X2 = pd.DataFrame(cleaned_comments_sentance)
Y2 = pd.DataFrame(list(data['y']))
X_train, X_test, Y_train, Y_test = train_test_split(X2, Y2, test_size=0.2)    
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=2)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=2)
second_model = compile_model(second_model, 'categorical_crossentropy', 40, 10, X_train, Y_train, X_test, Y_test)
third_model = compile_model(third_model, 'categorical_crossentropy', 40, 10, X_train, Y_train, X_test, Y_test)