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
import pandas as pd

import numpy as np
review = pd.read_csv("../input/yelp-reviews/yelp.csv")
import keras
MAX_NUM_WORDS = 100000

from keras.preprocessing.text import text_to_word_sequence, Tokenizer

tok = Tokenizer(num_words=MAX_NUM_WORDS)

tok.fit_on_texts(review.text.values)
tok.word_index
len(tok.word_index)
res = tok.texts_to_sequences(review.text)
len(res[1])
word_index_dict = tok.word_index
word_index_dict["the"]
index_word_dict = {v:k for k,v in word_index_dict.items()}
index_word_dict[30000]
[index_word_dict[i] for i in res[0]]
from keras.preprocessing.sequence import pad_sequences
MAX_SEQ_LEN=200

review_seq_padded = pad_sequences(

    res, maxlen=MAX_SEQ_LEN, dtype='int32', padding='pre', truncating='pre',

    value=0.0

)
review_seq_padded[0]
review.stars.value_counts()
y = np.where(review.stars.isin([4,5]),"good","not good")
from sklearn.model_selection import train_test_split
import numpy as np

y_num = np.where(y == "good",1,0)
X_train, X_test, y_train, y_test = train_test_split(review_seq_padded,y_num,random_state=100)
import tensorflow as tf



from tensorflow.keras import layers

from tensorflow.keras import losses

from tensorflow.keras import preprocessing

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D
tok.word_index
32*32
model = tf.keras.Sequential([

  layers.Input(X_train.shape[1]),

  layers.Embedding(len(tok.word_index) + 1, 32),

  layers.LSTM(16),

  layers.Dropout(0.2),

  layers.Dropout(0.2),

  layers.Dense(32),

  layers.Dense(1)])



model.summary()
# number of LSTM parameters

(16 + 32 + 1)*16*4



#num_params = [(num_units + input_dim + 1) * num_units] * 4

# 4 - for the 4 neural network layers in LSTM - W_forget, W_input, W_output, W_cell
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              optimizer=tf.keras.optimizers.Adam(1e-4),

              metrics=['accuracy'])



history = model.fit(

    X_train,y_train,

    validation_data=(X_test,y_test),

    epochs=10)


model = tf.keras.Sequential([

  layers.Embedding(len(tok.word_index) + 1, 32),

  layers.LSTM(16, return_sequences=True),

  layers.LSTM(8),

  layers.Dropout(0.2),

  layers.Dropout(0.2),

  layers.Dense(32),

  layers.Dense(1)])



model.summary()
model = tf.keras.Sequential([

  layers.Embedding(len(tok.word_index) + 1, 32),

  layers.Bidirectional(LSTM(16)),

  layers.Dropout(0.2),

  layers.Dropout(0.2),

  layers.Dense(32),

  layers.Dense(1)])



model.summary()
from gensim.models import KeyedVectors
EMBEDDING_FILE = "../input/gloveword2vec/glove.6B.50d.txt"
# load word2 vec

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE)
word2vec.word_vec("dog")
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(word2vec.word_vec("dog").reshape(1,-1),

                  word2vec.word_vec("cat").reshape(1,-1))
nrow_embed_matrix = len(tok.word_index)  + 1
EMBEDDING_DIM = 50

embedding_matrix = np.zeros((nrow_embed_matrix, EMBEDDING_DIM))
embedding_matrix.shape
for word, i in tok.word_index.items():

    if word in word2vec.vocab:

        embedding_matrix[i] = word2vec.word_vec(word)
len(tok.word_index)
import tensorflow as tf



from tensorflow.keras import layers

from tensorflow.keras import losses

from tensorflow.keras import preprocessing

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D
model = tf.keras.Sequential([

  layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],

                    weights=[embedding_matrix],trainable=True),

  layers.Bidirectional(LSTM(32)),

  layers.Dropout(0.2),

  layers.Dense(32),

  layers.Dense(1)])



model.summary()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              optimizer=tf.keras.optimizers.Adam(1e-4),

              metrics=['accuracy'])



history = model.fit(

    X_train,y_train,

    validation_data=(X_test,y_test),

    epochs=10)
import shutil

import os
os.makedirs("/kaggle/working/sentclass/pos")

os.makedirs("/kaggle/working/sentclass/neg")

for i in range(review.shape[0]):

    if (review.stars.iloc[i] == 4) | (review.stars.iloc[i] == 5):

        review_txt = review.text.iloc[i]

        file_name = "/kaggle/working/sentclass/pos/" + "pos_review_" + str(i) + ".txt"

        out_file = open(file_name, "w")

        out_file.writelines(review_txt)

        out_file.close()

    else:

        review_txt = review.text.iloc[i]

        file_name = "/kaggle/working/sentclass/neg/" + "neg_review_" + str(i) + ".txt"

        out_file = open(file_name, "w")

        out_file.writelines(review_txt)

        out_file.close()
len(os.listdir("/kaggle/working/sentclass/pos/"))
batch_size = 32

seed = 100



raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(

    '/kaggle/working/sentclass/', 

    batch_size=batch_size, 

    validation_split=0.2, 

    subset='training', 

    seed=seed)
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(

    '/kaggle/working/sentclass/', 

    batch_size=batch_size, 

    validation_split=0.2, 

    subset='validation', 

    seed=seed)
for text_batch, label_batch in raw_train_ds.take(1):

    for i in range(3):

        print("Review", text_batch.numpy()[i])

        print("Label", label_batch.numpy()[i])
print("Label 0 corresponds to", raw_train_ds.class_names[0])

print("Label 1 corresponds to", raw_train_ds.class_names[1])
max_features = 30000

sequence_length = 250



vectorize_layer = TextVectorization(

    max_tokens=max_features,

    output_mode='int',

    output_sequence_length=sequence_length)
import re

import string

# Make a text-only dataset (without labels), then call adapt

train_text = raw_train_ds.map(lambda x, y: x)

vectorize_layer.adapt(train_text)
def vectorize_text(text, label):

    text = tf.expand_dims(text, -1)

    return vectorize_layer(text), label
# retrieve a batch (of 32 reviews and labels) from the dataset

text_batch, label_batch = next(iter(raw_train_ds))

first_review, first_label = text_batch[0], label_batch[0]

print("Review", first_review)

print("Label", raw_train_ds.class_names[first_label])

print("Vectorized review", vectorize_text(first_review, first_label))
train_ds = raw_train_ds.map(vectorize_text)

val_ds = raw_val_ds.map(vectorize_text)
train_ds

embedding_dim = 16

model = tf.keras.Sequential([

  layers.Embedding(max_features + 1, embedding_dim),

  layers.LSTM(32),    

  layers.Dropout(0.2),

  layers.Dense(32),

  layers.Dropout(0.2),

  layers.Dense(1)])



model.summary()

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              optimizer=tf.keras.optimizers.Adam(1e-4),

              metrics=['accuracy'])
epochs = 10

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=epochs)
print("13 corresponds to - ",vectorize_layer.get_vocabulary()[1287])
len(vectorize_layer.get_vocabulary())