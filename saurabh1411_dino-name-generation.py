# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import keras

import tensorflow as tf

import os

from utils import *

import random
print(os.listdir('../input'))
data1= pd.read_csv('../input/dinosaur-island/dinos.txt',header=None)

data1.head()
data1.shape
print(os.listdir('../input/dinosaur-island'))
data = open('../input/dinosaur-island/dinos.txt').read()
#data = open('../input/dinos.txt').read()

data = data.lower()

chars = list(set(data))

data_size = len(data)

data_size, vocab_size = len(data), len(chars)

print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
data[0:21]
char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }

ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }

print(ix_to_char)
type(set(data))
# Build list of all dinosaur names (training examples).

with open("../input/dinosaur-island/dinos.txt") as f:

    examples = f.readlines()
examples[0:5]
examples = [x.lower().strip() for x in examples]

examples[0:5]
np.random.shuffle(examples)

m=len(examples)

m
'saurabh singh\n'.strip()
max_len = 30

num_classes = len(char_to_ix)



X = []

Y = []
for i in range(m):

    X_tmp = [char_to_ix[ch] for ch in examples[i]]

    X_tmp.extend([0] * (max_len - len(X_tmp)))

                 

    Y_tmp = X_tmp[1:] + [char_to_ix["\n"]]

    X.append(X_tmp)

    Y.append(Y_tmp)



X = np.array(X)

Y = np.array(Y)
print(X.shape)

print(Y.shape)
X = keras.utils.to_categorical(X, num_classes=num_classes, dtype='float32')

Y = keras.utils.to_categorical(Y, num_classes=num_classes, dtype='float32')

Y = np.swapaxes(Y,0,1)

print(X.shape)

print(Y.shape)
print(X[0].shape)

print(Y[0].shape)
from keras.models import Model, Sequential

from keras.layers import Input, Dense, Reshape, Lambda

from keras.layers import LSTM

from numpy import array

from keras.optimizers import Adam
n_a = 64

Tx = 30

reshapor = Reshape((1, num_classes))  

LSTM_cell = LSTM(n_a, return_state = True) 

densor = Dense(num_classes, activation='softmax')
def train_model(Tx, n_a, num_classes):

    X = Input(shape=(Tx, num_classes))

    a0 = Input(shape=(n_a,))

    c0 = Input(shape=(n_a,))

    

    a = a0

    c = c0

    

    outputs = []

    for t in range(Tx):

        x = Lambda(lambda x: X[:,t,:])(X)

        x = reshapor(x)

        a, _, c = LSTM_cell(x, initial_state=[a, c])

        out = densor(a) #num_classes

        outputs.append(out) # Tx, num_classes

    

    model = Model([X, a0, c0], outputs)

    return model
model = train_model(Tx, n_a, num_classes)
a0 = np.zeros((X.shape[0], n_a))

c0 = np.zeros((X.shape[0], n_a))

output = model.predict([X, a0, c0])
print(np.array(output).shape)
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
a0 = np.zeros((X.shape[0], n_a))

c0 = np.zeros((X.shape[0], n_a))

model.fit([X, a0, c0], list(Y), epochs=5)
x_initializer = np.zeros((1, 1, num_classes))

a_initializer = np.zeros((1, n_a))

c_initializer = np.zeros((1, n_a))
def generate_name_model(num_classes , n_a, Ty):

    # Define the input of your model with a shape 

    x0 = Input(shape=(1, num_classes))

    a0 = Input(shape=(n_a,), name='a0')

    c0 = Input(shape=(n_a,), name='c0')

    a = a0

    c = c0

    x = x0



    ### START CODE HERE ###

    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)

    outputs = []

    

    # Step 2: Loop over Ty and generate a value at every time step

    for t in range(Ty):

        

        # Step 2.A: Perform one step of LSTM_cell (≈1 line)

        a, _, c = LSTM_cell(x, initial_state=[a, c])

        

        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)

        out = densor(a)



        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)

        outputs.append(out)

        

    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)

    inference_model = Model([x0, a0, c0], outputs)

    

    ### END CODE HERE ###

    

    return inference_model
inference_model = generate_name_model(num_classes=num_classes, n_a = n_a, Ty = 30)
x_initializer = np.random.randn(1, 1, num_classes)

a_initializer = np.random.randn(1, n_a)

c_initializer = np.random.randn(1, n_a)

pred = inference_model.predict([x_initializer, a_initializer, c_initializer])

for item in pred:

    onehot = item[0]

    id = np.argmax(onehot)

    print(ix_to_char[id], end="")
# print(X[0])

for item in X[0]:

    id = np.argmax(item)

    print(ix_to_char[id], end="")