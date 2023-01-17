# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bs4 import BeautifulSoup

import re, datetime, functools

import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
tf.enable_eager_execution()
TRAIN_DATA_PATH = '/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv'

UNLABELD_TRAIN_DATA_PATH = '/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv'

TEST_DATA_PATH = '/kaggle/input/word2vec-nlp-tutorial/testData.tsv'
train_df = pd.read_csv(TRAIN_DATA_PATH, header=0, delimiter='\t', quoting=3)
train_df.head()
def review_to_words( raw_review ):

    # Function to convert a raw review to a string of words

    # The input is a single string (a raw movie review), and 

    # the output is a single string (a preprocessed movie review)

    #

    # 1. Remove HTML

    review_text = BeautifulSoup(raw_review).get_text() 

    #

    # 2. Remove non-letters        

    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    #

    # 3. Convert to lower case, split into individual words

    words = letters_only.lower().split()                             

    #

    # 4. In Python, searching a set is much faster than searching

    #   a list, so convert the stop words to a set

    stops = set(stopwords.words("english"))                  

    # 

    # 5. Remove stop words

    meaningful_words = [w for w in words if not w in stops]   

    #

    # 6. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join( meaningful_words ))  
clean_review = review_to_words( train_df["review"][0] )

print(clean_review)
train_df['preprocessed_review'] = [review_to_words(r) for r in train_df['review']]
max_features = 5000

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(train_df['preprocessed_review'])

train_data_features = tokenizer.texts_to_sequences(train_df['preprocessed_review'])

print(len(train_data_features))

train_data_features = tf.keras.preprocessing.sequence.pad_sequences(train_data_features, maxlen=150, padding='post')
print(train_data_features.shape)
Input = tf.keras.layers.Input

Dense = tf.keras.layers.Dense

GlobalAveragePooling1D = tf.keras.layers.GlobalAveragePooling1D

Dropout = tf.keras.layers.Dropout

Embedding = tf.keras.layers.Embedding

GRU = tf.keras.layers.CuDNNGRU

Bidirectional = tf.keras.layers.Bidirectional
def get_model():

    model = tf.keras.Sequential([

        Embedding(input_dim=max_features, output_dim=128),

        Bidirectional(GRU(32, return_sequences=True)),

        GlobalAveragePooling1D(),

        Dropout(0.03),

        Dense(20, activation='elu'),

        Dense(1, activation='sigmoid')

    ])



    model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



    model.summary()

    return model
model = get_model()
logdir = os.path.join("/tmp/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True),

    tensorboard_callback,

    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)

]
history = model.fit(train_data_features,

                    train_df['sentiment'], batch_size=100, epochs=20, validation_split=0.2, callbacks=callbacks)
df_test=pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv",header=0, delimiter="\t", quoting=3)

df_test.head()

df_test["review"]=df_test.review.apply(lambda x: review_to_words(x))

df_test["sentiment"] = df_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)

y_test = df_test["sentiment"]

list_sentences_test = df_test["review"]

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_te = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_test, maxlen=150)
prediction = model.predict(X_te)

y_pred = (prediction > 0.5)

from sklearn.metrics import f1_score, confusion_matrix

print('F1-score: {0}'.format(f1_score(y_pred, y_test)))

print('Confusion matrix:')

confusion_matrix(y_pred, y_test)