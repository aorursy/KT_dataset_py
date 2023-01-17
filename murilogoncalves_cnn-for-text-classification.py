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
# import the modules we'll need

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
from keras.preprocessing import sequence

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense, LSTM, Flatten, Conv1D

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight

from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
max_tokens = 256

seq_size = 1000
class Data():

    def __init__(self):

        self.__load(train=True)

        self.__load(train=False)

        self.setup_tokenizer()

        

    def __load(self, train=True):

        if train:

            self.train = pd.read_csv('/kaggle/input/train.csv')

            self.train = self.train.dropna()

            self.train = self.name_columns(self.train)

            self.clear_dataframe(self.train)

        else:

            self.test = pd.read_csv('/kaggle/input/test.csv')

            self.test = self.test.dropna()

            self.test = self.name_columns(self.test)

            self.clear_dataframe(self.test)

    

    def reload(self, train=True):

        self.__load(train, add_features)

    

    def name_columns(self, dataframe):

        dataframe.columns = ['category', 'title', 'content']

        return dataframe



    @staticmethod

    def clear_dataframe(dataframe):

        dataframe["title"] = dataframe['title'].str.lower()

        dataframe["content"] = dataframe['content'].str.lower()

        dataframe["full_content"] = dataframe[['title', 'content']].apply(lambda x: ' '.join(x), axis=1)



    def setup_tokenizer(self):

        self.tokenizer = Tokenizer(num_words=max_tokens, char_level=True)

        all_text = np.concatenate((self.train['full_content'].values, self.test['full_content'].values))

        self.tokenizer.fit_on_texts(all_text)

    

    def get_X(self, train=True):

        if train:

            dataframe = self.train

        else:

            dataframe = self.test

        text = dataframe['full_content'].values

        X = self.tokenizer.texts_to_sequences(text)  

        X = pad_sequences(X, maxlen=seq_size)

        return X

    

    def get_Y(self, train=True):

        if train:

            return pd.get_dummies(self.train['category'].values)

        else:

            return pd.get_dummies(self.test['category'].values)

        

data = Data()
from keras.layers import (Dense, concatenate, SpatialDropout1D,

                          Input,LSTM, Bidirectional, CuDNNLSTM, CuDNNGRU,

                          Activation,Conv1D,GRU, Dropout, GlobalMaxPooling1D, MaxPooling1D,

                          Embedding, GlobalAveragePooling1D, BatchNormalization)

from keras.models import Model

from keras import optimizers

from keras import regularizers

import keras.backend as K



def model():

    sequence_input = Input(shape=(seq_size,))

    emb = Embedding(seq_size, 100)(sequence_input)

    x= Conv1D(128, 5, activation='relu')(emb)

    x = MaxPooling1D(5)(x)

    x= Conv1D(128, 5, activation='relu')(emb)

    x = MaxPooling1D(5)(x)

    x= Conv1D(128, 5, activation='relu')(emb)

    x = MaxPooling1D(35)(x)

    x = Flatten()(x)

    x = Dropout(0.2)(Dense(128, activation='relu')(x))

    preds = Dense(4, activation='softmax')(x)

    model = Model(sequence_input, preds)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    model.summary()

    return model
base_model = model()

hist = base_model.fit(data.get_X(True), data.get_Y(True), 

              validation_data =(data.get_X(False), data.get_Y(False)),

              batch_size=512, nb_epoch = 10,  verbose = 1)

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])
import seaborn as sn

pred = 1 + np.argmax(base_model.predict(data.get_X(False)), axis=1)
data.test['pred'] = pred
cm = confusion_matrix(data.test['category'], data.test['pred'])

plt.imshow(cm, cmap=plt.cm.Blues)