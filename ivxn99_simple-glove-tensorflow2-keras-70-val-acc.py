import os

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, GlobalMaxPooling1D, Embedding

from tensorflow.keras import Sequential

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



data_path = '/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv'

data = pd.read_csv(data_path, sep='\t')

data.head()
import re

def get_clean_text(x):

    x = re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', '', x) 

    #regex to remove to emails(above)

    x = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x)

    #regex to remove URLs

    x = re.sub('RT', "", x)

    #substitute the 'RT' retweet tags with empty spaces

    x = re.sub('[^A-Z a-z]+', '', x)

    return x

data['verified_reviews'] = data['verified_reviews'].apply(lambda x: get_clean_text(x))
print("Max review length: ",data['verified_reviews'].map(len).max())

print("Min review length: ",data['verified_reviews'].map(len).min())

print("Average tweet length: ", data['verified_reviews'].map(len).mean())



chars = sorted(list(set(data['verified_reviews'])))



plt.figure(figsize=(10,6))

sns.countplot(x=data['rating'])
X = data.verified_reviews

y = data.rating.map({1:0, 2:1, 3:2, 4:3, 5:4})



train_size = int(len(data) * 0.8)

X_train, y_train = X[:train_size], y[:train_size]

X_test, y_test = X[train_size:], y[train_size:]
!wget http://nlp.stanford.edu/data/glove.6B.zip

!unzip -q glove.6B.zip
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization



vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=150)

text_ds = tf.data.Dataset.from_tensor_slices(X_train).batch(128)

vectorizer.adapt(text_ds)
#We can retrieve the computed vocabulary like this:

#Let's print the top 5 words

vectorizer.get_vocabulary()[:5]



#Here's a dict mapping words to their indices:

voc = vectorizer.get_vocabulary()

word_index = dict(zip(voc, range(len(voc))))
path_to_glove_file = '/content/glove.6B.100d.txt'



embeddings_index = {}

with open(path_to_glove_file) as f:

  for line in f:

    word, coefs = line.split(maxsplit=1)

    coefs = np.fromstring(coefs, "f", sep=" ")

    embeddings_index[word] = coefs



print("Found %s word vectors." % len(embeddings_index))
num_tokens = len(voc)

embedding_dim = 100

hits = 0

misses = 0



embedding_matrix = np.zeros((num_tokens, embedding_dim))

for word, i in word_index.items():

  embedding_vector = embeddings_index.get(word)

  if embedding_vector is not None:

    #words not found in embedding index will be all-zeros

    #This includes the rpresentation for "padding" and "OOV"

    embedding_matrix[i] = embedding_vector

    hits += 1

  else:

    misses +=1

print("Converted %d words (%d misses)" % (hits, misses))
embedding_layer = Embedding(num_tokens, embedding_dim, 

                            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),

                            trainable=False)
int_sequences_input = tf.keras.Input(shape=(None,), dtype="int64")

embedded_sequences = embedding_layer(int_sequences_input)

x = Conv1D(128, 5, activation="relu")(embedded_sequences)

x = MaxPooling1D(5)(x)

x = Conv1D(128, 5, activation="relu")(x)

x = MaxPooling1D(5)(x)

x = Conv1D(128, 5, activation="relu")(x)

x = GlobalMaxPooling1D()(x)

x = Dense(128, activation="relu")(x)

x = Dropout(0.5)(x)

preds = Dense(5, activation="softmax")(x)

model = tf.keras.Model(int_sequences_input, preds)

model.summary()
x_train = vectorizer(np.array([[s] for s in X_train])).numpy()

x_test = vectorizer(np.array([[s] for s in X_test])).numpy()



y_train = np.array(y_train)

y_val = np.array(y_test)
x_train = vectorizer(np.array([[s] for s in X_train])).numpy()

x_test = vectorizer(np.array([[s] for s in X_test])).numpy()



y_train = np.array(y_train)

y_val = np.array(y_test)
checkpoint_path = '/content/checkpoint'

callback = tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_path,

    save_weights_only=True,

    monitor='val_acc',

    save_best_only=True

)



model.compile(

    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"]

)

model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test), callbacks=[callback])