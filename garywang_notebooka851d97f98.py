# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

train_dat = train_data.as_matrix()

test_dat = test_data.as_matrix()



# Any results you write to the current directory are saved as output.
# pull the label and data, and format the training input output

Labels = train_dat[:,0]

train_Y = np.zeros((len(Labels), 10))

train_Y[np.arange(len(train_Y)), Labels] = 1

train_X = train_dat[:,1:]



test_X = test_dat
# here we do the keras import and build using Keras API

from keras.models import Model # import model building API

from keras.layers import Dense, Dropout, BatchNormalization, Input



# building our Neural net

inputs = Input(shape=(784,)) # create input

hidden1 = BatchNormalization()(inputs) # apply batch norm to input

hidden1 = Dense(100, activation = 'relu')(hidden1) # 200 unit hidden layer 1, with relu activation

hidden1 = Dropout(0.3)(hidden1) # apply dropout, at 0.3

hidden2 = Dense(100, activation = 'relu')(hidden1) # 200 unit hidden layer 2, with relu activation

hidden2 = Dropout(0.3)(hidden2) # apply 0.3 dropout

output = Dense(10, activation = 'softmax')(hidden2) # apply final softmax output layer



Net = Model(input = inputs, output = output) # create our Net with Model API

Net.compile(optimizer = 'adam',

           loss = 'categorical_crossentropy',

           metrics = ['accuracy'])



# after our net compiled show some net statistics

Net.summary()
# now we can proceed to train our neuarl network

# Note the input is fed in as shape [batch_size, input_dim]

Net.fit(train_X, train_Y, batch_size = 64, nb_epoch = 4)
# lastly we predict our output

predict_label = Net.predict(test_X)

prediction = predict_label.argmax(axis=1)



# and save as prediction file

pred = np.column_stack((np.asarray(range(1,len(prediction)+1)), prediction))

submission = pd.DataFrame(pred, columns=['ImageId', 'Label'])

submission.to_csv('submission_logreg.csv', index=False)