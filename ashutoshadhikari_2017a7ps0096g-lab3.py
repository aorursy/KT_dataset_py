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

import matplotlib.pyplot as plt
import random
import os

train = pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/nlp_train.csv')
test = pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/_nlp_test.csv')

#X_train, X_val, y_train, y_val = train_test_split(train['tweet'], train['offensive_language'], test_size = 0.2, random_state = 42)
X_train = train['tweet']
X_test = test['tweet']
y_train = train['offensive_language']
df_submit = pd.DataFrame()
df_submit['tweet'] = test['tweet']
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))
num_words = 20000
max_features = 20000
tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')
tokenizer.fit_on_texts(X_train.values)
X_train = tokenizer.texts_to_sequences(X_train.values)
X_test = tokenizer.texts_to_sequences(X_test.values)
word_index = tokenizer.word_index

max_length_of_text = 200
X_train = pad_sequences(X_train, maxlen=max_length_of_text)
X_test = pad_sequences(X_test, maxlen=max_length_of_text)
embedding_dim = 300
# Get embeddings
embeddings_index = {}
f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt')
for line in f:
    values = line.rstrip().rsplit(' ', embedding_dim)
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
# Build embedding matrix
embedding_matrix = np.zeros((max_features, embedding_dim))
for word, i in word_index.items():
    if i == max_features:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
batch_size = 256 
num_epochs = 50

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]
embed_dim = 300 #Change to observe effects
batch_size = 32

model3 = Sequential()
model3.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_length_of_text, trainable=True))
# Add Convolutional layer
model3.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model3.add(MaxPooling1D(3))
model3.add(GlobalMaxPooling1D())
model3.add(BatchNormalization())
model3.add(Dropout(0.3))
model3.add(Dense(128, activation = "relu"))
model3.add(Dropout(0.3))
model3.add(Dense(32, activation = "relu"))
model3.add(Dropout(0.3))
model3.add(Dense(1, activation = 'linear'))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model3.compile(loss=root_mean_squared_error, optimizer=adam, metrics=['accuracy'])
model3.summary()

hist = model3.fit(X_train, y_train, batch_size=batch_size, callbacks = callbacks_list, epochs=num_epochs, validation_split=0.2, shuffle=True, verbose=2)

y_pred = model3.predict(X_test)
for i in y_pred:
    if i < 0.5:
        i = 0.0
    elif i < 1.5:
        i = 1.0
    elif i < 2.5:
        i = 2.0
    else:
        i = 3.0

df_submit['offensive_language'] = y_pred
df_submit.to_csv('submit_glove_3.csv', index = False)
model3.save_weights('model3.h5')
df_submit.to_csv('submit_glove_3.csv', index = False)
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
