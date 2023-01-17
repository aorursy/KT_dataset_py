# Importing dependencies numpy and keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
# load text
filename = "../input/Macbeth.txt"

text = (open(filename).read()).lower()

# mapping characters with integers
unique_chars = sorted(list(set(text)))

char_to_int = {}
int_to_char = {}

for i, c in enumerate (unique_chars):
    char_to_int.update({c: i})
    int_to_char.update({i: c})
print(unique_chars)
# preparing input and output dataset
X = []
Y = []

for i in range(0, len(text) - 50, 1):
    sequence = text[i:i + 50] # save the first 49 characters as input
    label = text[i + 50] # the 50th  character as output
    X.append([char_to_int[char] for char in sequence])
    Y.append(char_to_int[label])
# reshaping, normalizing and one hot encoding
X_modified = np.reshape(X, (len(X), 50, 1))
X_modified = X_modified / float(len(unique_chars))
Y_modified = np_utils.to_categorical(Y)
X_modified
# defining the LSTM model
model = Sequential()
model.add(LSTM(300, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True)) # input layer
model.add(Dropout(0.2))
model.add(LSTM(300)) # hidden layer
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax')) # output layer

model.compile(loss='categorical_crossentropy', optimizer='adam')
# fitting the model
model.fit(X_modified, Y_modified, epochs=5, batch_size=30)
# picking a random seed
start_index = np.random.randint(0, len(X)-1)
new_string = X[start_index]
print(new_string)
# generating characters
for i in range(100):
    x = np.reshape(new_string, (1, len(new_string), 1))
    x = x / float(len(unique_chars))

    #predicting
    pred_index = np.argmax(model.predict(x, verbose=0))
    char_out = int_to_char[pred_index]
    seq_in = [int_to_char[value] for value in new_string]
    print(char_out)

    new_string.append(pred_index)
    new_string = new_string[1:len(new_string)]
print(new_string)