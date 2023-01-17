# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import tensorflow as tf





# Any results you write to the current directory are saved as output.
SEED = 42

tf.random.set_random_seed(SEED)

np.random.seed(SEED)

VOCAB_SIZE = 100_000  # Limit on the number vocabulary size used for tokenization

MAX_SEQUENCE_LENGTH = 50  # Sentences will be truncated/padded to this length
df_twitter = pd.read_csv("../input/sentiment140/training.1600000.processed.noemoticon.csv",encoding="latin1", header=None)

df_twitter = df_twitter.rename(columns={

                 0:"sentiment",

                 1:"id",

                 2:"time",

                 3:"query",

                 4:"username",

                 5:"text"

             })

df_twitter = df_twitter[["sentiment","text"]]
df_twitter["sentiment_label"] = df_twitter["sentiment"].map({0: 0, 4: 1})
import re

from tensorflow.keras.preprocessing import text

from tensorflow.keras.preprocessing.sequence import pad_sequences



class TextPreprocessor(object):

    

    def __init__(self, vocab_size, max_sequence_length):

        self._vocab_size = vocab_size

        self._max_sequence_length = max_sequence_length

        

        #imported so the functions are avaialble in the pickle without importing

        self._tokenizer = text.Tokenizer(num_words=self._vocab_size, oov_token = '<OOV>')

        self._pad_sequences = pad_sequences

        self._re_sub = re.sub

        

    def _clean_line(self, text):

        text = self._re_sub(r"http\S+", "<url>", text)

        text = self._re_sub(r"@[A-Za-z0-9]+", "<user>", text)

#         text = self._re_sub(r"#[A-Za-z0-9]+", "", text)

        text = text.replace("RT","")

        text = text.lower()

        text = text.strip()

        return text

    

    def fit(self, text_list):        

        # Create vocabulary from input corpus.

        text_list_cleaned = [self._clean_line(txt) for txt in text_list]

        self._tokenizer.fit_on_texts(text_list)



    def transform(self, text_list):        

        # Transform text to sequence of integers

        text_list = [self._clean_line(txt) for txt in text_list]

        seq = self._tokenizer.texts_to_sequences(text_list)

        padded_text_sequence = self._pad_sequences(seq, maxlen=self._max_sequence_length)

        return padded_text_sequence
processor = TextPreprocessor(5, 5)

processor.fit(['hello machine learning','test'])

processor.transform(['hello machine learning',"lol"])
from sklearn.model_selection import train_test_split



sents = df_twitter.text

labels = np.array(df_twitter.sentiment_label)



# Train and test split

print('Splitting Data')

X_train, X_test, y_train, y_test = train_test_split(sents, labels, test_size=0.2, random_state = SEED)



print('X_train.shape', X_train.shape)



processor = TextPreprocessor(VOCAB_SIZE, MAX_SEQUENCE_LENGTH)

processor.fit(X_train)



# Preprocess the data

print('Processing data')

X_train = processor.transform(X_train)

print('X_train.shape', X_train.shape)



print('Processing data')

X_test = processor.transform(X_test)



with open('./preprocessor.pkl', 'wb') as f:

    pickle.dump(processor, f)
print('Test Processor')

print('Size of Word Index:', len(processor._tokenizer.word_index))

processor.transform(['Hello World'])
LEARNING_RATE=.001

EMBEDDING_DIM=50

FILTERS=64

DROPOUT_RATE=0.5

POOL_SIZE=2

NUM_EPOCH=25

BATCH_SIZE=256

KERNEL_SIZES=[2,4,8]



def create_model(

        embedding_matrix,

        filters=FILTERS, 

        kernel_sizes=KERNEL_SIZES, 

        dropout_rate=DROPOUT_RATE, 

        pool_size=POOL_SIZE):

    

    # Input layer

    model_input = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')



    # Embedding layer

    x = tf.keras.layers.Embedding(

        input_dim=(embedding_matrix.shape[0]),

        output_dim=embedding_matrix.shape[1],

        input_length=MAX_SEQUENCE_LENGTH,

        weights=[embedding_matrix]

    )(model_input)



    x = tf.keras.layers.Dropout(rate = dropout_rate)(x)



    # Convolutional block

    conv_blocks = []

    for kernel_size in kernel_sizes:

        conv = tf.keras.layers.Convolution1D(

                        filters=filters,

                        kernel_size=kernel_size,

                        padding="valid",

                        activation="relu",

                        strides=1

                    )(x)

        conv = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)

        conv = tf.keras.layers.Flatten()(conv)

        conv_blocks.append(conv)

        

    x = tf.keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]



    x = tf.keras.layers.Dropout(rate = dropout_rate)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)

    model_output = tf.keras.layers.Dense(1, activation="sigmoid")(x)



    model = tf.keras.models.Model(model_input, model_output)

    

    return model
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')



def get_embedding_matrix(processor, vocab_size, embeddings_index):

    word_index = processor._tokenizer.word_index

    nb_words = min(vocab_size, len(word_index))

    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))



    for word, i in word_index.items():

        if i >= VOCAB_SIZE: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

EMBEDDINGS_INDEX = dict(get_coefs(*o.strip().split()) for o in open("../input/glove-global-vectors-for-word-representation/glove.twitter.27B.50d.txt","r",encoding="utf8"))
EMBEDDING_MATRIX = get_embedding_matrix(processor = processor, vocab_size = VOCAB_SIZE, embeddings_index = EMBEDDINGS_INDEX)
print('EM shape', EMBEDDING_MATRIX.shape)
model = create_model(embedding_matrix=EMBEDDING_MATRIX)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#keras train

history = model.fit(

    X_train, 

    y_train, 

    epochs=NUM_EPOCH, 

    batch_size=BATCH_SIZE,

    validation_data=(X_test, y_test),

    verbose=2,

    callbacks=[

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor='val_acc',

            min_delta=0.005,

            patience=3,

            factor=0.5

        ),

        tf.keras.callbacks.EarlyStopping(

            monitor='val_loss',

            min_delta=0.005, 

            patience=5,  

            mode='auto',

            restore_best_weights = True

        ),

        tf.keras.callbacks.History()

    ]

)
model.save('keras_saved_model.h5')
print('Y Train Mean', y_train.mean())

print('Y Test Mean', y_test.mean())
import matplotlib.pyplot as plt

plt.plot(history.epoch, history.history['loss'], label='train loss')

plt.plot(history.epoch, history.history['val_loss'], label='val loss')

plt.legend()

plt.show()
plt.plot(history.epoch, history.history['acc'], label='train acc')

plt.plot(history.epoch, history.history['val_acc'], label='val acc')

plt.legend()

plt.show()