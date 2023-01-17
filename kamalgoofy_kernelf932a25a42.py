# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, LSTM

import numpy as np 



# Any results you write to the current directory are saved as output.
NUM_CHARACTERS = 27

SEQ_LENGTH = 3



text = 'machine learning'

def char2int(c):

    if c == ' ':

        return 26

    return ord(c) - ord('a')



def int2char(n):

    if n == 26:

        return ' '

    return chr(ord('a') + n)



def str2int(s):

    ret = []

    for c in s:

        ret.append(char2int(c))

    return ret



def int2str(l):

    ret = ''

    l = np.array(l, int)

    for n in l:

        ret += int2char(n)

    return ret





print(str2int("kamal"))

print(int2str([10,0,12,0,11]))

train_x = []

train_y = []



text_int = str2int(text)



EYE = np.eye(NUM_CHARACTERS)



for seq in range(len(text_int) -SEQ_LENGTH-1):

    sequence = text_int[seq:seq+SEQ_LENGTH+1]

    

    if len(sequence) != SEQ_LENGTH+1:

        break

    seq_x = []

    for i in range(SEQ_LENGTH):

        this_int = sequence[i]

        this_onehot = EYE[this_int]

        seq_x.append(this_onehot)

    train_x.append(seq_x)

    next_int = sequence[SEQ_LENGTH]

    next_onehot = EYE[next_int]

    train_y.append(next_onehot)

        

        

train_x = np.array(train_x, dtype=int)

train_y = np.array(train_y, dtype=int)

    

print(train_x.shape)

print(train_y.shape)
model = Sequential()



model.add(LSTM(128, input_shape=(SEQ_LENGTH, NUM_CHARACTERS), activation = 'relu'))



model.add(Dense(32, activation='relu'))



model.add(Dense(27, activation='softmax'))



optimizer =tf.keras.optimizers.Adam(lr=0.001, decay=1e-5)



model.compile(loss='mean_squared_error', optimizer=optimizer)



model.summary()

history = model.fit(x=train_x, y=train_y, epochs=100, batch_size=16, verbose=1)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
text = 'mac'

index = 0

while len(text) < 100:

    prev = text[index:index+3]

    index += 1

    prev = str2int(prev)

    prev_onehot = [EYE[i] for i in prev]

    next_char = model.predict(np.reshape(prev_onehot, [1,3,NUM_CHARACTERS]))

    next_char = np.argmax(next_char ,axis=1)

    next_char = int2char(next_char[0])

    text += next_char

    

print("The following text is written by RNN: \n" + text)