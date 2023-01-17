import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

import logging

plt.style.use('fivethirtyeight')



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM, GRU

from tensorflow.keras.models import Sequential

from tensorflow.keras.datasets import imdb
print("reading data..")

data = pd.read_csv('/kaggle/input/imdb-review-dataset/imdb_master.csv',encoding='ISO-8859-1')
class IMDBSentiMentAnalysis:

    

    def __init__(self, data, maxlen=100, num_words=10000):

        

        self.model = None

        self.history = None

        self.data = data

        self.maxlen = maxlen

        self.num_words = num_words

        

    def process_data(self):

        print("processing data...")

        data = self.data[self.data.label!='unsup']

        sns.countplot(x='label',data=data)

        data['out'] = data['label']

        

        data['out'][data.out=='neg']=0

        data['out'][data.out=='pos']=1

        # Another way data['out'] = data['out'].map({1:'pos',0:'neg'})

        np.unique(data.out)

        #data['out'] = data['label'].map({1:'pos',0:'neg'})

        

        req_data = data[['review','out']]



        self.texts = np.array(req_data.review)

        self.labels = np.array(req_data.out)

        self.convert_data_to_padded_sequence()

            

            

    def convert_data_to_padded_sequence(self):

        print("Converting data to Sequences")

        # num_words: Top No. of words to be tokenized. Rest will be marked as unknown or ignored.

        tokenizer = Tokenizer(num_words=self.num_words) 

        

        # tokenizing based on "texts". This step generates the word_index and map each word to an integer other than 0.

        tokenizer.fit_on_texts(self.texts)

        

        # generating sequence based on tokenizer's word_index. Each sentence will now be represented by combination of numericals

        # Example: "Good movie" may be represented by [22, 37]

        seq = tokenizer.texts_to_sequences(self.texts)

        

        self.word_index = tokenizer.word_index

        # padding each numerical representation of sentence to have fixed length.



        self.padded_seq = np.array(pad_sequences(seq,maxlen=self.maxlen))

        print("Data converted to Sequences...")

        

    

    

    def plot_model_output(self):

        history = self.history

        epochs = self.epochs

        plt.figure()

        plt.plot(range(epochs,),history.history['loss'],label = 'training_loss')

        plt.plot(range(epochs,),history.history['val_loss'],label = 'validation_loss')

        plt.legend()

        plt.figure()

        plt.plot(range(epochs,),history.history['acc'],label = 'training_accuracy')

        plt.plot(range(epochs,),history.history['val_acc'],label = 'validation_accuracy')

        plt.legend()

        plt.show()



    def init_model(self, model = None, gru=False):

        

        if model is None:

            print("Initialising default model")

            model = Sequential()

            embedding = Embedding(self.num_words, 32, input_length = self.maxlen, name='embedding')

            model.add(embedding)

            if gru:

                model.add(GRU(32))

            else:

                model.add(LSTM(32))

            model.add(Flatten())

            model.add(Dense(1,activation='sigmoid'))

            self.model = model

        else:

            print("Initialising model passed")

            self.model = model



        return self.model.summary()



    def run_the_model(self,optimizer = 'rmsprop', epochs = 10, validation_split=0.2):

        

        self.model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])

        self.epochs = epochs



        self.history = self.model.fit(self.padded_seq,np.asarray(self.labels).astype(np.uint8),epochs=epochs,validation_split=validation_split)

        

        self.plot_model_output()

    
print("Initialising IMDB object")



imdb_deep_learning = IMDBSentiMentAnalysis(data,)



imdb_deep_learning.process_data()
imdb_deep_learning.init_model(gru=True)
imdb_deep_learning.run_the_model(epochs = 10)
imdb_deep_learning.init_model(gru=False)

imdb_deep_learning.run_the_model(epochs = 10)