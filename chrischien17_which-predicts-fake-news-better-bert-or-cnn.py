# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras

from keras.preprocessing.text import Tokenizer



from keras.layers import Dropout, Dense,Input,Embedding,Flatten, MaxPooling1D, Conv1D

from keras.models import Sequential,Model

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics

from keras.preprocessing.sequence import pad_sequences

from sklearn.datasets import fetch_20newsgroups

from keras.layers.merge import Concatenate

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



from keras.layers import Dropout, Dense,Input,Embedding,Flatten, MaxPooling1D, Conv1D

from keras.models import Sequential,Model

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

import pandas as pd

from sklearn import metrics

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.datasets import fetch_20newsgroups

from keras.layers.merge import Concatenate



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



from keras.backend import concatenate
# for bert model

!pip install ktrain

import ktrain

from ktrain import text
df = pd.read_csv('/kaggle/input/a-fake-news-dataset-around-the-syrian-war/FA-KES-Dataset.csv',encoding='latin1')

df.info()
# df['article_title'].apply(lambda x: ' ') is used to create a space between two column text

texts = np.array(df['article_title'] + df['article_title'].apply(lambda x: ' ') + df['article_content'])

target = df['labels']

# texts[0], target[0]
target_names = np.unique(target).tolist()

target_names
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

texts, target, test_size=0.3, random_state=42)



X_train = X_train.tolist()

X_test = X_test.tolist()

y_train = y_train.tolist()

y_test = y_test.tolist()
(X_train,  y_train), (X_test, y_test), preproc = text.texts_from_array(x_train=X_train, y_train=y_train,

                                                                       x_test=X_test, y_test=y_test,

                                                                       class_names=target_names,

                                                                       preprocess_mode='bert',

                                                                       maxlen=350, 

                                                                       max_features=35000)
# you can disregard the deprecation warnings arising from using Keras 2.2.4 with TensorFlow 1.14.

model = text.text_classifier('bert', train_data=(X_train, y_train), preproc=preproc)

learner = ktrain.get_learner(model, train_data=(X_train, y_train), batch_size=6)
learner.fit_onecycle(2e-5, 5)
learner.validate(val_data=(X_test, y_test), class_names=target_names)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(texts, target, test_size=0.3, random_state=42)
# tokenization

MAX_NB_WORDS = 3000

tokenizer = Tokenizer(num_words=MAX_NB_WORDS) # based on the word frequency, num_words-1 words will be kept

tokenizer.fit_on_texts(texts)
# word indexing

word_index = tokenizer.word_index

print("Data type of word index: {} and length of dictionary {}".format(type(word_index),

                                                                       len(word_index)) )
# vectorization

sequences_train = tokenizer.texts_to_sequences(X_train)

sequences_test = tokenizer.texts_to_sequences(X_test)

print("Sequence: ", sequences_train[0])
# padding

MAX_SEQUENCE_LENGTH = 1600

texts_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH) # default value truncate the previous sequence

texts_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH) # default value truncate the previous sequence

print("Padding Result: ", texts_train[0])
# assignment of word embedding

## access pretrained word embedding file

path='/kaggle/input/gloveicg/glove/Glove/glove.6B.300d.txt'

embeddings_index = {}

f = open(path, encoding="utf8")

for line in f:

    values = line.split()

    word = values[0]

    try:

        coefs = np.asarray(values[1:], dtype='float32')

    except:

        pass

    embeddings_index[word] = coefs

f.close()



## build word embedding matrix based on word index

EMBEDDING_DIM = 300

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))



for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        if len(embedding_matrix[i]) !=len(embedding_vector):

            print("could not broadcast input array from shape",str(len(embedding_matrix[i])),

                                 "into shape",str(len(embedding_vector))," Please make sure your"

                                 " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")

            exit(1)

        embedding_matrix[i] = embedding_vector



embedding_matrix.shape # words, vector length
print("Number of samples, Max sequence length, embedding dim")

len(df), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
embedding_layer = Embedding(len(word_index) + 1,

                                EMBEDDING_DIM,

                                weights=[embedding_matrix],

                                input_length=MAX_SEQUENCE_LENGTH,

                                trainable=True)
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

# embedded_sequences = embedding_layer(sequence_input)

# flat_1 = Flatten()(embedded_sequences)

# flat_1.shape

# embedded_sequences.shape
dropout=0.5

nclasses = 2



# input layer and embedding layer

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='input_embedding')

embedded_sequences = embedding_layer(sequence_input)

flat_1 = Flatten(name='flatten_1')(embedded_sequences)

# dense layer and dropout layer

l_dense = Dense(512, activation='relu', name='dense_layer_1')(flat_1)

l_dropout = Dropout(dropout)(l_dense)



### prediction

preds = Dense(nclasses, activation='softmax', name='prediction_layer')(l_dropout)



model = Model([sequence_input], preds)

model.compile(loss='sparse_categorical_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])
from keras.utils.vis_utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.fit(texts_train, y_train,

              batch_size=128,

              epochs=5,

          validation_data =(texts_test, y_test),

          verbose=True)
loss, accuracy = model.evaluate(texts_test, y_test, verbose=0)

print("Loss value: {}, accuracy:{}".format(loss, accuracy))