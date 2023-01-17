# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/winemag-data-130k-v2.csv')
dataset.head()
dataset = dataset.iloc[:, [2,12]]
dataset.head()
dataset = dataset.drop_duplicates()
before_check = len(dataset)

dataset = dataset.dropna(axis = 0, how = 'any')
after_check = len(dataset)

print('{} row(s) removed from dataset due to NaN values being present'.format(before_check - after_check))
descriptions = np.array(dataset.iloc[:,0])
labels = np.array(dataset.iloc[:,-1])
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categories = 'auto')
labels = ohe.fit_transform(labels.reshape(-1,1))

num_classes = labels.shape[1]
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def tokenize(description):
    description = description.lower()
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(description)
    
    # remove stop words
    tokens = [word for word in tokens if not word in stop_words]
    
    # stem words
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    
    # remove digits
    tokens = [word for word in tokens if not word.isdigit()]
    
    return tokens

tokens = [tokenize(description) for description in descriptions]
from keras.preprocessing import text, sequence

max_features = 10000
max_len = 100

tokenizer = text.Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(tokens)
sequences = tokenizer.texts_to_sequences(tokens)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = sequence.pad_sequences(sequences, maxlen = max_len)
num_samples = round(len(data) * 0.8)

x_train = data[:num_samples]
y_train = labels[:num_samples]

x_test = data[num_samples:]
y_test = labels[num_samples:]

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Embedding(max_features, 128, input_length = max_len))
    model.add(layers.Conv1D(128, 7, activation = 'relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation = 'softmax'))
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
    
    return model
from keras import callbacks

callbacks = [callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)]
model = build_model()
history = model.fit(x_train, y_train,
                    epochs = 20,
                    batch_size = 32,
                    validation_split = 0.2,
                    callbacks = callbacks)
model.evaluate(x_test, y_test)
