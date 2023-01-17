import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


import re
import numpy as np
### necessary functions from the keras library
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model
import keras
import random
import pickle

# Read text from himu.txt file 
bangla_text = open('../input/himu.txt').read()
print('আমাদের অরিজিনাল টেক্সট এ ' + str(len(bangla_text)) + ' বর্ণ আছে')
# Printing 2000 words
bangla_text[:2000]
def remove_tabs(text):
    
    text = text.replace('\n',' ') 
    text = text.replace('\t',' ')
    text = text.replace('\r',' ')
    text = text.replace('\u200d',' ')
    
    text = re.sub(' +',' ',text)
    
    return text

bangla_text = remove_tabs(bangla_text)
# After remove new line and tabs
bangla_text[:2000]
def remove_symbols(txt):
    
    chars = ['$', '!', '@', '?', '%', '/', '*', '-', '&', '(', ')', '"', "'", ',', ';', ':',
             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0','‘','।','’','—', 
              '১', '২', '৩', '\u200c', '–', '“', '”', '…',
              'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's',
             't', 'u', 'w', 'y', 'z', '।', ',', '?','I', 'K', 'R', 'T', 'Y'] 
    for c in chars:
        if c in txt:
            txt = txt.replace(c, " ")

    # যদি অতিরিক্ত স্পেস থাকে তাহলে একটা স্পেস এ রিপ্লেস করবে। 
    txt = txt.replace('  ',' ')
    
    return txt
    
bangla_text = remove_symbols(bangla_text)
# After cleaning
bangla_text[:2000]
# টোটাল ইউনিক ক্যারেক্টার গণনা 
chars = sorted(list(set(bangla_text)))

print ("এই corpus এ আছে সর্বমোট " +  str(len(bangla_text)) + " টি ক্যারেক্টার ")
print ("এই corpus এ আছে সর্বমোট " +  str(len(chars)) + " টি ইউনিক ক্যারেক্টার")
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = [text[i:i+window_size] for i in range(0, len(text)-window_size, step_size)]
    outputs = [text[i] for i in range(window_size, len(text), step_size)]    
    return inputs,outputs
# run your text window-ing function 
window_size = 100
step_size = 5
inputs, outputs = window_transform_text(bangla_text,window_size,step_size)
# print out a few of the input/output pairs to verify that we've made the right kind of stuff to learn from
print('input = ' + inputs[1])
print('output = ' + outputs[1])
print('--------------')
print('input = ' + inputs[302])
print('output = ' + outputs[302])
# print out the number of unique characters in the dataset
chars = sorted(list(set(bangla_text)))
print ("this corpus has " +  str(len(chars)) + " unique characters")
print ('and these characters are ')
print (chars)
# this dictionary is a function mapping each unique character to a unique integer
chars_to_indices = dict((c, i) for i, c in enumerate(chars))  # map each unique character to unique integer

# this dictionary is a function mapping each unique integer back to a unique character
indices_to_chars = dict((i, c) for i, c in enumerate(chars))  # map each unique integer back to unique character
# transform character-based input/output into equivalent numerical versions
def encode_io_pairs(text,window_size,step_size):
    # number of unique chars
    chars = sorted(list(set(text)))
    num_chars = len(chars)
    
    # cut up text into character input/output pairs
    inputs, outputs = window_transform_text(text,window_size,step_size)
    
    # create empty vessels for one-hot encoded input/output
    X = np.zeros((len(inputs), window_size, num_chars), dtype=np.bool)
    y = np.zeros((len(inputs), num_chars), dtype=np.bool)
    
    # loop over inputs/outputs and tranform and store in X/y
    for i, sentence in enumerate(inputs):
        for t, char in enumerate(sentence):
            X[i, t, chars_to_indices[char]] = 1
        y[i, chars_to_indices[outputs[i]]] = 1
        
    return X,y
window_size = 100
step_size = 5
X,y = encode_io_pairs(bangla_text,window_size,step_size)
def create_rrn_model():

    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model
model = create_rrn_model()
model.summary()
# a small subset of our input/output pairs
Xtrain = X[:,:,:]
ytrain = y[:,:]
def training(model, Xtrain, ytrain, batch_size=500, epochs=50):
    
    model.fit(Xtrain, ytrain, batch_size=500, epochs=epochs,verbose = 1)
    
    # মডেল সেভ
    model.save('model.h5')
training(model, Xtrain, ytrain,epochs=5)
with open('history', 'wb') as file_pi:
        pickle.dump(model.history, file_pi)
# function that uses trained model to predict a desired number of future characters
def predict_next_chars(model,input_chars,num_to_predict):     
    # create output
    predicted_chars = ''
    for i in range(num_to_predict):
        # convert this round's predicted characters to numerical input    
        x_test = np.zeros((1, window_size, 61))
        #print(x_test.shape)
        #x_test.reshape(1, 100, 28)
        for t, char in enumerate(input_chars):
            
            x_test[0, t, chars_to_indices[char]] = 1.

        # make this round's prediction
        test_predict = model.predict(x_test,verbose = 0)[0]

        # translate numerical prediction back to characters
        r = np.argmax(test_predict)                           # predict class of each test input
        d = indices_to_chars[r] 

        # update predicted_chars and input
        predicted_chars+=d
        #print(r)
        input_chars+=d
        input_chars = input_chars[1:]
    return predicted_chars
# get an appropriately sized chunk of characters from the text
start_inds = [200, 500, 800, 1200]

# load in weights
#model.load_weights('model_weights/best_RNN_small_textdata_weights.hdf5')
new_model = load_model('model.h5')
for s in start_inds:
    start_index = s
    input_chars = bangla_text[start_index: start_index + window_size]

    # use the prediction function
    predict_input = predict_next_chars(new_model,input_chars,num_to_predict = 100)

    # print out input characters
    print('------------------')
    input_line = 'input chars = ' + '\n' +  input_chars + ' "' + '\n'
    print(input_line)

    # print out predicted characters
    line = 'predicted chars = ' + '\n' +  predict_input + '"' + '\n'
    print(line)
