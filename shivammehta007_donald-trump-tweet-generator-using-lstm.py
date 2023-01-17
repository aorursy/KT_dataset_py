from __future__ import print_function



import keras

from keras.layers import Input, Embedding, LSTM, Dense, Dropout

from keras.utils import np_utils

from keras.models import Model

from keras.models import Sequential

from keras.optimizers import adam, adagrad, adadelta, rmsprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import EarlyStopping

from keras.regularizers import L1L2



import matplotlib.pyplot as plt

from sklearn.model_selection import ParameterGrid, train_test_split



import os

import numpy as np

import pandas as pd





os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Read Data

folder_name = '../input'

filename = os.path.join(folder_name, 'task3_corpus.csv')

# filename = 'task3_corpus.csv'

file_type = 'csv'
def read_data(filename, file_type):

    if file_type == 'csv':

        data = pd.read_csv(filename)

        data = data['text']

        

    return data
df = read_data(filename, file_type)
text = '\n'.join([row for row in df])
layers = [( 'LSTM', 150), ('Dropout', 0.2), ('LSTM', 120)]

count = [x for x,_ in layers].count('LSTM')

print(count)

class ModelFormer:

    def __init__(self):

        self.x = []

        self.y = []

        self.tokenizer = Tokenizer()

        self.best_model = Sequential()

        self.best_accuracy = 0

        self.best_parameters = {}

        

    def fit_data(self, text):

        self.original_corpus = text

        self.corpus = self.original_corpus.lower().split('\n')

        self.tokenizer.fit_on_texts(self.corpus)

        self.word_count = len(self.tokenizer.word_index) + 1

        input_sequences = []

        for line in self.corpus:

            tokens = self.tokenizer.texts_to_sequences([line])[0]

            for i in range(1, len(tokens)):

                n_grams_sequence = tokens[:i+1]

                input_sequences.append(n_grams_sequence)

        

        input_sequences = self.pad_input_sequences(input_sequences)

        

        x_data, y_data = input_sequences[:,:-1], input_sequences[:,-1]

        y_data = np_utils.to_categorical(y_data, num_classes=self.word_count)

        

        return x_data, y_data

              

    def pad_input_sequences(self,input_sequences):

        max_sequence_length = max([len(sentence) for sentence in input_sequences])

        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

        return input_sequences

    

    def fit(self, x_data, y_data , layers= [( 'LSTM', 150), ('Dropout', 0.2), ('LSTM', 120)], activation='tanh', optimizer='adam', lr=0.01, epochs=20):

        self.model = Sequential()

        

        self.x_data = x_data

        self.y_data = y_data

        x_train, x_val, y_train, y_val = train_test_split(self.x_data, self.y_data)

        

        

        self.model.add(Embedding(self.word_count, 10, input_length=len(x_data[0]) ))

        count_lstm_retn_flag = [x for x,_ in layers].count('LSTM') - 1



        for layer,value in layers:

            if layer == 'LSTM':

                if count_lstm_retn_flag:

                    count_lstm_retn_flag -= 1

                    return_sequences = True 

                else:

                    return_sequences = False

                self.model.add(LSTM(value, activation=activation, return_sequences=return_sequences))

            if layer == 'Dropout':

                self.model.add(Dropout(value))

        

        self.model.add(Dense(self.word_count, activation='softmax'))

        if optimizer == 'adam':

            optimizer = adam(lr=lr)

        elif optimizer == 'adadelta':

            optimizer = adadelta(lr=lr)

        elif optimizer == 'rmsprop':

            optimizer = rmsprop(lr=lr)

            

            

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.model.summary()

        

        fit_summary = self.model.fit(x_train, y_train, epochs=epochs, verbose=0, validation_data=(x_val, y_val), batch_size=50)

        if fit_summary.history['acc'][-1] > self.best_accuracy:

            self.best_model = self.model

            self.best_accuracy = fit_summary.history['acc'][-1]

            self.best_parameters = (layers, activation, optimizer, lr, epochs)

        

        return fit_summary

        

        
m = ModelFormer()

X, Y= m.fit_data(text)

x_train , x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)
print(len(x_train), len(y_train), len(x_test), len(y_test))
# Define Grid Search with Parameter Grid
hyperparameters = { 'layers': [ [( 'LSTM', 200), ('Dropout', 0.2)], [( 'LSTM', 200), ('Dropout', 0.2), ('LSTM', 400), ('Dropout', 0.2) ]], 

                     'activation': ['tanh'],

                     'optimizer' : [ ('adam', 0.01 ), ('adam', 0.001 ) , ('adadelta', 1 ), ('rmsprop', 0.1 )],

                     'epochs' : [50]

                   }
combinations = list(ParameterGrid(hyperparameters))

combinations
fit_summary_array = []
for combination in combinations:

    print('Current Combination : {}'.format(combination))

    fit_summary_array.append(m.fit(x_train, y_train, layers=combination['layers'], activation=combination['activation'], optimizer=combination['optimizer'][0], lr=combination['optimizer'][1], epochs=combination['epochs']))
print('Best Accuracy : {}, with best Parameters : {}'.format(m.best_accuracy*100, m.best_parameters))
# Generate Sentences : 

def generate_n_sentences(n=5):

    final_sentences = []

    for _ in range(n):

        prediction = x_test[np.random.randint(len(x_test))]

        prediction = np.delete(prediction, 0)

        first_prediction = m.best_model.predict_classes([x_test[0].reshape(1,54)])

        prediction = np.append(prediction,first_prediction)

        for _ in range(5):

            next_prediction = m.best_model.predict_classes(prediction.reshape(1,54))

            prediction = np.delete(prediction, 0)

            prediction = np.append(prediction,next_prediction)







        output_word = ""

        for i in prediction:

            if i:

                for word,index in m.tokenizer.word_index.items():

                    if index == i:

                        output_word += word + ' '

                        break



        final_sentences.append(output_word)

    return final_sentences
generate_n_sentences(10)