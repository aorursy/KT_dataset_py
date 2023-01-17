from __future__ import print_function, division

from builtins import range



import os

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from keras.layers import Dense, Input, GlobalMaxPooling1D, Embedding, Lambda, Concatenate,LSTM, Bidirectional

from keras.models import Model

from keras.optimizers import Adam

from sklearn.metrics import roc_auc_score

import keras.backend as K



print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.info()
train = train.as_matrix()  # we make each row into matrix to have width and height like an image
np.random.shuffle(train)  # shuffle data to make it more random
# X has 784 pixels so we make it into 28 by 28 square (28^2 = 784)

X = train[:, 1:].reshape(-1, 28, 28) / 255  # data is from 0 to 255 so we divide by 255 to make it between [0,1]

Y = train[:, 0]

M = 15

input = Input(shape=(28, 28)) # feed in image of size 28x28



rnn1 = Bidirectional(LSTM(M, return_sequences=True))   # LSTM layer with 15 neurons

x1 = rnn1(input)   # size is 28 x 28 x 30 (2M)

x1 = GlobalMaxPooling1D()(x1)   # after maxpooling size is 28 x 30 (basically each columns becomes 1 number wich is most prominent)



rnn2 = Bidirectional(LSTM(M, return_sequences=True))

# here we want to transpose the image so width and height change place (turn image 90 degrees)

# a lambda layer takes any function and can perform it on all elements of the input

# t is input tensors inputs, K.permute_dimensions is the function

# permute_dimensions : takes the input and a tuple of 3d dimensions for input, here we swapped place of 1 and 2

shapeShifter = Lambda(lambda t: K.permute_dimensions(t, pattern=(0, 2, 1)))

x2 = shapeShifter(input)

x2 = rnn2(x2)

x2 = GlobalMaxPooling1D()(x2)



# add two rnn outputs together on horizontal axis

concatenator = Concatenate(axis = 1)

x = concatenator([x1, x2])  # 28 x 30 + 28 x 30 = 28 x 60



# final dense layer

output = Dense(10, activation='softmax')(x)



model = Model(inputs=input, outputs=output)



# our Y is numbers 0 to 9 and we don't want to use one-hot encoding so we use sparse_categorical_crossentropy

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
r = model.fit(X, Y, batch_size=32, epochs=10, validation_split=0.3)
test = test.as_matrix()

X = test.reshape(-1, 28, 28) / 255
res = model.predict(X, batch_size=32, verbose=1)
final = pd.read_csv("../input/sample_submission.csv")



f = []

for seq in res:

    num = np.argmax(seq)

    f.append(num)

       

my_submission = pd.DataFrame({"ImageId": final.ImageId, "Label": f})



# Explicitly include the argument index=False to prevent pandas from adding another column in our csv file.

my_submission.to_csv('submission.csv', index=False)
