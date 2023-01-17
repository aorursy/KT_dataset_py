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
from collections import Counter
import time
import os
import numpy as np
import pandas as pd
import re
import itertools
from tqdm import tqdm
from tqdm import  tqdm_notebook
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import random
from tensorflow.keras.preprocessing.text import Tokenizer 
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Dropout
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D , BatchNormalization
from tensorflow.keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, LSTM
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras import backend

import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import os, re, csv, math, codecs

#imports and train-test split

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, LeakyReLU, Flatten, Activation, MaxPool2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.layers import Lambda, SeparableConv2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import mean_squared_error

# Import package

from string import punctuation
from collections import defaultdict
# from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Activation, LSTM, Lambda
from tensorflow.keras.layers import Concatenate as concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.layers.convolutional import Conv1D
from tensorflow.keras.layers import GlobalAveragePooling1D
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *

# Define constants and parameters

import matplotlib.pyplot as plt
import random
import os

train = pd.read_csv('/kaggle/input/nnfl-lab-4/train.csv')
test = pd.read_csv('/kaggle/input/nnfl-lab-4/test.csv')
train.head()
test.head()
# Define constants and parameters

Max_Sequence_Length = 200
Max_Num_Words = 100000 
Embedding_Dim = 300
Validation_Split_Ratio = 0.1

Num_Lstm = np.random.randint(175, 275)
Num_Dense = np.random.randint(100, 150)
Rate_Drop_Lstm = 0.15 + np.random.rand() * 0.25
Rate_Drop_Dense = 0.15 + np.random.rand() * 0.25

act_f = 'relu'
re_weight = False # whether to re-weight classes to fit the 17.4% share in test set

# Create word embedding dictionary from 'glove.840B.300d.txt'

embeddings_index = {}
f = open('/kaggle/input/fasttext-wikinews/wiki-news-300d-1M.vec')

# for line in tqdm(f):
for line in f:
    values = line.split()
    # word = values[0]
    word = ''.join(values[:-300])   
    # coefs = np.asarray(values[1:], dtype='float32')
    coefs = np.asarray(values[-300:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    text = text.lower().split()
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        text = [w for w in text if not w in stop_words]
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    return(text)
train_texts_1 = [] 
train_texts_2 = []
train_labels = []

df_train = pd.read_csv('/kaggle/input/nnfl-lab-4/train.csv', encoding='utf-8')
df_train = df_train.fillna('empty')
train_q1 = df_train.Sentence1.values
train_q2 = df_train.Sentence2.values
train_labels = df_train.Class.values

for text in train_q1:
    train_texts_1.append(text_to_wordlist(text, remove_stopwords=False, stem_words=False))
    
for text in train_q2:
    train_texts_2.append(text_to_wordlist(text, remove_stopwords=False, stem_words=False))
test_texts_1 = []
test_texts_2 = []
test_ids = []

df_test = pd.read_csv('/kaggle/input/nnfl-lab-4/test.csv', encoding='utf-8')
df_test = df_test.fillna('empty')
test_q1 = df_test.Sentence1.values
test_q2 = df_test.Sentence2.values
test_ids = df_test.ID.values

for text in test_q1:
    test_texts_1.append(text_to_wordlist(text, remove_stopwords=False, stem_words=False))
    
for text in test_q2:
    test_texts_2.append(text_to_wordlist(text, remove_stopwords=False, stem_words=False))
# Tokenize words in all sentences
tokenizer = Tokenizer(num_words=Max_Num_Words)
tokenizer.fit_on_texts(train_texts_1 + train_texts_2 + test_texts_1 + test_texts_2)

train_sequences_1 = tokenizer.texts_to_sequences(train_texts_1)
train_sequences_2 = tokenizer.texts_to_sequences(train_texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index

# pad all train with Max_Sequence_Length
train_data_1 = pad_sequences(train_sequences_1, maxlen=Max_Sequence_Length)
train_data_2 = pad_sequences(train_sequences_2, maxlen=Max_Sequence_Length)

# pad all test with Max_Sequence_Length
test_data_1 = pad_sequences(test_sequences_1, maxlen=Max_Sequence_Length)
test_data_2 = pad_sequences(test_sequences_2, maxlen=Max_Sequence_Length)

num_words = min(Max_Num_Words, len(word_index))+1

embedding_matrix = np.zeros((num_words, Embedding_Dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
emb_layer = Embedding(
    input_dim=num_words,
    output_dim=Embedding_Dim,
    weights=[embedding_matrix],
    input_length=Max_Sequence_Length,
    trainable=False
)    

# 1D convolutions that can iterate over the word vectors
conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

# Define inputs
seq1 = Input(shape=(Max_Sequence_Length,), dtype='int32')
seq2 = Input(shape=(Max_Sequence_Length,), dtype='int32')

 # Run inputs through embedding
emb1 = emb_layer(seq1)
emb2 = emb_layer(seq2)

# Run through CONV + GAP layers
conv1a = conv1(emb1)
glob1a = GlobalAveragePooling1D()(conv1a)
conv1b = conv1(emb2)
glob1b = GlobalAveragePooling1D()(conv1b)

conv2a = conv2(emb1)
glob2a = GlobalAveragePooling1D()(conv2a)
conv2b = conv2(emb2)
glob2b = GlobalAveragePooling1D()(conv2b)

conv3a = conv3(emb1)
glob3a = GlobalAveragePooling1D()(conv3a)
conv3b = conv3(emb2)
glob3b = GlobalAveragePooling1D()(conv3b)

conv4a = conv4(emb1)
glob4a = GlobalAveragePooling1D()(conv4a)
conv4b = conv4(emb2)
glob4b = GlobalAveragePooling1D()(conv4b)

conv5a = conv5(emb1)
glob5a = GlobalAveragePooling1D()(conv5a)
conv5b = conv5(emb2)
glob5b = GlobalAveragePooling1D()(conv5b)

conv6a = conv6(emb1)
glob6a = GlobalAveragePooling1D()(conv6a)
conv6b = conv6(emb2)
glob6b = GlobalAveragePooling1D()(conv6b)

mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])

merge =concatenate([diff, mul])

# The MLP that determines the outcome
x = Dropout(0.2)(merge)
x = BatchNormalization()(x)

x = Dense(300, activation='relu')(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)

pred = Dense(1, activation='sigmoid')(x)

# model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
model2 = Model(inputs=[seq1, seq2], outputs = pred)
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model2.summary()
early_stopping =EarlyStopping(monitor='val_loss', patience=10)
bst_model_path = 'model.h5' 
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)


hist = model2.fit([train_data_1, train_data_2], train_labels, \
        validation_split = 0.1, \
        epochs=20, batch_size=128, shuffle=True, \
         callbacks=[early_stopping, model_checkpoint])

model2.load_weights(bst_model_path) # sotre model parameters in .h5 file
bst_val_score = min(hist.history['val_loss'])
preds = model2.predict([test_data_1, test_data_2], batch_size=128, verbose=1)
for i in preds:
    if i[0] < 0.5:
        i[0] = int(0)
    else:
        i[0] = int(1)
df_submit = pd.read_csv('/kaggle/input/nnfl-lab-4/test.csv')
df_submit['Class'] = preds.astype(int)
df_submit.drop(columns = ['Sentence1', 'Sentence2'], inplace = True)
df_submit.to_csv('submit_cnn_2.csv', index = False)
model2.save_weights('model_cnn_2.h5')
df = df_submit
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df)
