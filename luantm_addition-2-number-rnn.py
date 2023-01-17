import tensorflow as tf
import numpy as np 
import pandas as pd
import os
import random
class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One-hot encode given string C.
        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.
        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)
training_data = 50000
digits = 3
chars = '0123456789+ '
ctable = CharacterTable(chars)
question_max_len = digits * 2 + 1
answer_max_len = digits + 1

# random 0 -> 10^digits - 1()
def create_data():
    x = []
    y = []
    for i in range(training_data):
        a = random.randint(0, 10**digits - 1)
        b = random.randint(0, 10**digits - 1)
        
        question = '{}+{}'.format(a, b)
        question = question + ' '*(question_max_len - len(question))
        answer = str(a + b)
        answer = answer + ' '*(answer_max_len - len(answer))
        
        x.append(question)
        y.append(answer)
    return x, y
    
data_x, data_y = create_data()
print(data_x[0])
print(data_y[0])
#Vectorization
x = np.zeros((training_data, question_max_len, len(chars)), dtype=np.bool)
y = np.zeros((training_data, answer_max_len, len(chars)), dtype=np.bool)
for i, sentence in enumerate(data_x):
    x[i] = ctable.encode(sentence, question_max_len)
for i, sentence in enumerate(data_y):
    y[i] = ctable.encode(sentence, answer_max_len)
print(x.shape)
print(y.shape)
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, Dense, TimeDistributed
model = Sequential()
model.add(LSTM(128, input_shape=x.shape[1:]))
model.add(RepeatVector(answer_max_len))
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributed(
    Dense(len(chars), activation='softmax')
))
          
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
          
model.fit(x, y, epochs=59, validation_split=0.2)
# for iteration in range(1, 200):
#     print()
#     print('-' * 50)
#     print('Iteration', iteration)
#     model.fit(x_train, y_train,
#               batch_size=BATCH_SIZE,
#               epochs=1,
#               validation_data=(x_val, y_val))
#     # Select 10 samples from the validation set at random so we can visualize
#     # errors.
#     for i in range(10):
#         ind = np.random.randint(0, len(x_val))
#         rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
#         preds = model.predict_classes(rowx, verbose=0)
#         q = ctable.decode(rowx[0])
#         correct = ctable.decode(rowy[0])
#         guess = ctable.decode(preds[0], calc_argmax=False)
#         print('Q', q[::-1] if REVERSE else q, end=' ')
#         print('T', correct, end=' ')
#         if correct == guess:
#             print(colors.ok + '☑' + colors.close, end=' ')
#         else:
#             print(colors.fail + '☒' + colors.close, end=' ')
#         print(guess)