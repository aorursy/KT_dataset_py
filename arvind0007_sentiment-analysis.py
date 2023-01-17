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
# Basic packages

import pandas as pd 

import numpy as np

import re

import collections

import matplotlib.pyplot as plt



# Packages for data preparation

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer

from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder



# Packages for modeling

from keras import models

from keras import layers

from keras import regularizers



from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers.embeddings import Embedding

import keras

import tensorflow as tf
df = pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')

df = df.reindex(np.random.permutation(df.index))  

df = df[['text', 'airline_sentiment']]

df.head()
def remove_mentions(input_text):

        return re.sub(r'@\w+', '', input_text)

def remove_urls(input_text):

        return re.sub(r"http\S+", "", input_text)

def remove_punctuations(input_text):

        return re.sub(r'[^\w\d\s\']+', '', input_text)

def remove_numbers(input_text):

        return re.sub(r"[0-9]"," ",input_text)

df.text = df.text.apply(remove_mentions).apply(remove_urls).apply(remove_numbers).apply(remove_punctuations)

df.head()

X_train, X_test, y_train, y_test = train_test_split(df.text, df.airline_sentiment, test_size=0.1, random_state=37)

print('# Train data samples:', X_train.shape[0])

print('# Test data samples:', X_test.shape[0])

assert X_train.shape[0] == y_train.shape[0]

assert X_test.shape[0] == y_test.shape[0]
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer()

X_train_tokenizer={}

for x in X_train.index:

    X_train_tokenizer[x]=tknzr.tokenize(X_train[x].lower())
X_test_tokenizer={}

for x in X_test.index:

    X_test_tokenizer[x]=tknzr.tokenize(X_test[x].lower())
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) 



for x in X_train.index:

    X_train_tokenizer[x] = [w for w in X_train_tokenizer[x] if not w in stop_words] 
for x in X_test.index:

    X_test_tokenizer[x] = [w for w in X_test_tokenizer[x] if not w in stop_words] 
from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences
embedded_sentences=[]

padded_sentences={}

labels=[]

for x in X_train.index:

    sentence=''

    if(y_train[x]=="positive"):

        labels.append([1,0,0])

    elif(y_train[x]=="negative"):

        labels.append([0,0,1])

    elif (y_train[x]=="neutral"):

        labels.append([0,1,0])

    else:

        print(x)

    for s in X_train_tokenizer[x]:

        sentence+=s

        sentence+=" "

    embedded_sentences.append(one_hot(sentence, 12000))

    #padded_sentences[x]=pad_sequences(embedded_sentences[x], 21, padding='post')

    

    #print(embedded_sentences[x])#,padded_sentences[x] )



padded_sentences=pad_sequences(embedded_sentences,31,padding='post')









embedded_sentences_test=[]

padded_sentences_test={}

labels_test=[]

for x in X_test.index:

    sentence=''

    if(y_test[x]=="positive"):

        labels_test.append([1,0,0])

    elif(y_test[x]=="negative"):

        labels_test.append([0,0,1])

    elif (y_test[x]=="neutral"):

        labels_test.append([0,1,0])

    else:

        print(x)

    for s in X_test_tokenizer[x]:

        sentence+=s

        sentence+=" "

    embedded_sentences_test.append(one_hot(sentence, 12000))

    #padded_sentences[x]=pad_sequences(embedded_sentences[x], 21, padding='post')

    

    #print(embedded_sentences[x])#,padded_sentences[x] )



padded_sentences_test=pad_sequences(embedded_sentences_test,31,padding='post')









from keras import backend as K



def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
class MultiHeadSelfAttention(layers.Layer):

    def __init__(self, embed_dim, num_heads=8):

        super(MultiHeadSelfAttention, self).__init__()

        self.embed_dim = embed_dim

        self.num_heads = num_heads

        if embed_dim % num_heads != 0:

            raise ValueError(

                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"

            )

        self.projection_dim = embed_dim // num_heads

        self.query_dense = layers.Dense(embed_dim)

        self.key_dense = layers.Dense(embed_dim)

        self.value_dense = layers.Dense(embed_dim)

        self.combine_heads = layers.Dense(embed_dim)



    def attention(self, query, key, value):

        score = tf.matmul(query, key, transpose_b=True)

        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)

        scaled_score = score / tf.math.sqrt(dim_key)

        weights = tf.nn.softmax(scaled_score, axis=-1)

        output = tf.matmul(weights, value)

        return output, weights



    def separate_heads(self, x, batch_size):

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))

        return tf.transpose(x, perm=[0, 2, 1, 3])



    def call(self, inputs):

        # x.shape = [batch_size, seq_len, embedding_dim]

        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)

        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)

        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)

        query = self.separate_heads(

            query, batch_size

        )  # (batch_size, num_heads, seq_len, projection_dim)

        key = self.separate_heads(

            key, batch_size

        )  # (batch_size, num_heads, seq_len, projection_dim)

        value = self.separate_heads(

            value, batch_size

        )  # (batch_size, num_heads, seq_len, projection_dim)

        attention, weights = self.attention(query, key, value)

        attention = tf.transpose(

            attention, perm=[0, 2, 1, 3]

        )  # (batch_size, seq_len, num_heads, projection_dim)

        concat_attention = tf.reshape(

            attention, (batch_size, -1, self.embed_dim)

        )  # (batch_size, seq_len, embed_dim)

        output = self.combine_heads(

            concat_attention

        )  # (batch_size, seq_len, embed_dim)

        return output

    

class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):

        super(TransformerBlock, self).__init__()

        self.att = MultiHeadSelfAttention(embed_dim, num_heads)

        self.ffn = keras.Sequential(

            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]

        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)

        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)

        self.dropout2 = layers.Dropout(rate)



    def call(self, inputs, training):

        attn_output = self.att(inputs)

        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)

    

embed_dim = 256  # Embedding size for each token

num_heads = 64  # Number of attention heads

ff_dim = 64



inputs = layers.Input(shape=(31,))

embedding_layer = Embedding(12000, 256, trainable=True)(inputs)

transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

x = transformer_block(embedding_layer)

x = layers.GlobalAveragePooling1D()(x)

x = layers.Dropout(0.1)(x)

x = layers.Dense(32, activation="relu")(x)

x = layers.Dropout(0.1)(x)

x = layers.Dense(16, activation="relu")(x)

x = layers.Dropout(0.1)(x)

x = layers.Dense(8, activation="relu")(x)

x = layers.Dropout(0.1)(x)

outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m])
inputs = keras.Input(shape=(31,), dtype="int32")

# Embed each integer in a 128-dimensional vector

x = layers.Embedding(12000, 128)(inputs)

# Add 2 bidirectional LSTMs

x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

x = layers.Bidirectional(layers.LSTM(64))(x)

# Add a classifier

outputs = layers.Dense(3, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
max_features=12000

embedding_dim=128

inputs = tf.keras.Input(shape=(None,), dtype="int64")



# Next, we add a layer to map those vocab indices into a space of dimensionality

# 'embedding_dim'.

x = layers.Embedding(max_features, embedding_dim)(inputs)

x = layers.Dropout(0.5)(x)



# Conv1D + global max pooling

x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)

x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)

x = layers.GlobalMaxPooling1D()(x)



# We add a vanilla hidden layer:

x = layers.Dense(128, activation="relu")(x)

x = layers.Dropout(0.5)(x)



# We project onto a single unit output layer, and squash it with a sigmoid:

predictions = layers.Dense(3, activation="sigmoid", name="predictions")(x)



model = tf.keras.Model(inputs, predictions)



# Compile the model with binary crossentropy loss and an adam optimizer.

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",f1_m])
labels=np.array(labels)

print(labels.shape)
labels_test=np.array(labels_test)
model.fit(padded_sentences, labels, epochs=5, verbose=1,batch_size=64,validation_data=(padded_sentences_test,labels_test))
model.evaluate(padded_sentences_test,labels_test)
all_words=[]

max_length=0

for x in X_train.index:

    if max_length<len(X_train_tokenizer[x]):

        max_length=len(X_train_tokenizer[x])

    for w in X_train_tokenizer[x]:

        all_words.append(w)

unique_words = set(all_words)

print(len(unique_words),max_length)
all_words=[]

max_length=0

for x in X_t.index:

    if max_length<len(X_test_tokenizer[x]):

        max_length=len(X_test_tokenizer[x])

    for w in X_test_tokenizer[x]:

        all_words.append(w)

unique_words = set(all_words)

print(len(unique_words),max_length)
max_len=0

for x in X_train.index:

    if max_len<len(X_train[x].split()):

        max_len=len(X_train[x].split())

    #print(x,X_train[x],y_train[x])

print(max_len)