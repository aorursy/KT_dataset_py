# Importing Libraries
import pandas as pd

import numpy as np

import seaborn as sns
import os 



# This is to get the directory that the program 

# is currently running in. 

dir_path = os.path.dirname(os.path.realpath('./')) 



for root, dirs, files in os.walk(dir_path):

    for file in files:

        if file.endswith('.csv'):

            print(root+'/'+str(file))
# Importing tensorflow

np.random.seed(42)

import tensorflow as tf

tf.set_random_seed(42)
# Configuring a session

session_conf = tf.ConfigProto(

    intra_op_parallelism_threads=1,

    inter_op_parallelism_threads=1

)
# Import Keras

from keras import backend as K

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)
# Importing libraries

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers.core import Dense, Dropout
# Initializing parameters

epochs = 30

batch_size = 16

n_hidden = 32
i=X_train=np.load("../input/X_train"+ '.npy')

print(i.shape)

i=Y_train=np.load("../input/Y_train"+ '.npy')

print(i.shape)

i=X_test=np.load("../input/X_test"+ '.npy')

print(i.shape)

i=Y_test=np.load("../input/Y_test"+ '.npy')

print(i.shape)
print(i.shape)
timesteps = len(X_train[0])

input_dim = len(X_train[0][0])

#n_classes = _count_classes(Y_train)



print(timesteps)

print(input_dim)

print(len(X_train))
from keras.models import Sequential

from keras.utils import np_utils

from keras.layers.core import Dense, Activation, Dropout
from __future__ import print_function



from hyperopt import Trials, STATUS_OK, tpe, rand

from keras.layers.core import Dense, Dropout, Activation

from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential

from keras.utils import np_utils

from sklearn.metrics import accuracy_score

from hyperas import optim

from hyperas.distributions import choice, uniform

from keras import optimizers

from keras.models import Sequential

from keras.utils import np_utils

from keras.layers.core import Dense, Activation, Dropout

from keras.utils import np_utils

from keras.layers.core import Dense, Activation, Dropout

from keras.layers import Flatten

import numpy as np

# Activities are the class labels

# It is a 6 class classification

ACTIVITIES = {

    0: 'WALKING',

    1: 'WALKING_UPSTAIRS',

    2: 'WALKING_DOWNSTAIRS',

    3: 'SITTING',

    4: 'STANDING',

    5: 'LAYING',

}



# Utility function to print the confusion matrix

def confusion_matrix(Y_true, Y_pred):

    Y_true = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_true, axis=1)])

    Y_pred = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_pred, axis=1)])



    return pd.crosstab(Y_true, Y_pred, rownames=['True'], colnames=['Pred'])
from keras.layers import RNN

from keras.layers import SimpleRNN

model = Sequential()

# Configuring the parameters

#cell = MinimalRNNCell(32)

model.add(Dense(n_hidden, input_shape=(128, 9)))

model.add(SimpleRNN(32,activation='relu',return_sequences=True))

#model.add(RNN(32,return_sequences=True))

#model.add(Dropout(0.15))

# Adding a dense output layer with sigmoid activation



#model.add(Dense(n_hidden, input_shape=(timesteps, input_dim),activation='sigmoid'))

model.add(Flatten())

model.add(Dense(6, activation='sigmoid'))

model.summary()
p=[rmsprop,sgd,adagrad,adadelta,adam,adamax,nadam]
from keras import optimizers



# All parameter gradients will be clipped to

# a maximum norm of 1.

rmsprop = optimizers.RMSprop(clipnorm=1)

sgd=optimizers.SGD(clipnorm=1)

adagrad=optimizers.Adagrad(clipnorm=1)

adadelta=optimizers.Adadelta(clipnorm=1)

adam=optimizers.Adam(clipnorm=1)

adamax=optimizers.Adamax(clipnorm=1)

nadam=optimizers.Nadam(clipnorm=1)

p=[rmsprop,sgd,adagrad,adadelta,adam,adamax,nadam]

model.compile(loss='categorical_crossentropy', optimizer=nadam,metrics=['accuracy'])



print("Training...")

model.fit(X_train, Y_train, nb_epoch=10, verbose=2)

score = model.evaluate(X_test, Y_test)

loss = model.evaluate(X_test, Y_test, verbose=0)

print("the loss is ",loss)
sns.heatmap(confusion_matrix(Y_test, model.predict(X_test)),annot=True)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

model = Sequential()

# Configuring the parameters

model.add(Dense(n_hidden, input_shape=(128, 9)))

model.add(Bidirectional(LSTM(30, return_sequences=True)))

model.add(Dropout(0.15))

# Adding a dense output layer with sigmoid activation



#model.add(Dense(n_hidden, input_shape=(timesteps, input_dim),activation='sigmoid'))

model.add(Flatten())

model.add(Dense(6, activation='sigmoid'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

print("Training...")

model.fit(X_train, Y_train, nb_epoch=10, verbose=2)

score = model.evaluate(X_test, Y_test)

loss = model.evaluate(X_test, Y_test, verbose=0)

print("the loss is ",loss)

sns.heatmap(confusion_matrix(Y_test, model.predict(X_test)),annot=True)
y_true=Y_test

y_pred=model.predict(X_test)

#(np.array

print(y_pred)

import keras

keras.metrics.categorical_accuracy(y_true, y_pred)
import seaborn as sns

print("the loss is ",loss)

sns.heatmap(confusion_matrix(Y_test, model.predict(X_test)),annot=True)
model.metrics_names
import seaborn as sns

print("the loss is ",loss)

sns.heatmap(confusion_matrix(Y_test, model.predict(X_test)),annot=True)
model = Sequential()

# Configuring the parameters

model.add(Dense(n_hidden, input_shape=(timesteps, input_dim)))

model.add(Activation('relu'))

# Adding a dropout layer

model.add(Dropout(0.1))

# Adding a dense output layer with sigmoid activation



#model.add(Dense(n_hidden, input_shape=(timesteps, input_dim),activation='sigmoid'))

model.add(Flatten())

model.add(Dense(6, activation='sigmoid'))

model.summary()



# we'll use categorical xent for the loss, and RMSprop as the optimizer

for i in p:

    model.compile(loss='categorical_crossentropy', optimizer=i)

    model.summary()

    print("Training...")

    model.fit(X_train, Y_train, nb_epoch=10, batch_size=16, validation_split=0.1,  verbose=2)

    loss = model.evaluate(X_test, Y_test, verbose=0)

    print("the loss is ",loss)

#sns.heatmap(confusion_matrix(Y_test, model.predict(X_test)),annot=True)

model = Sequential()

# Configuring the parameters

model.add(Dense(n_hidden, input_shape=(timesteps, input_dim)))

model.add(Activation('relu'))

# Adding a dropout layer

model.add(Dropout(0.1))

# Adding a dense output layer with sigmoid activation

model.add(Activation('relu'))

# Adding a dropout layer

model.add(Dropout(0.1))

#model.add(Dense(n_hidden, input_shape=(timesteps, input_dim),activation='sigmoid'))

model.add(Flatten())

model.add(Dense(6, activation='sigmoid'))

model.summary()





for i in p:

    model.compile(loss='categorical_crossentropy', optimizer=i)

    model.summary()

    print("Training...")

    model.fit(X_train, Y_train, nb_epoch=10, batch_size=16, validation_split=0.1,  verbose=2)

    loss = model.evaluate(X_test, Y_test, verbose=0)

    print("the loss is ",loss)

# we'll use categorical xent for the loss, and RMSprop as the optimizer

#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#model.summary()

#print("Training...")

#model.fit(X_train, Y_train, nb_epoch=10, batch_size=16, validation_split=0.1,  verbose=2)

#score = model.evaluate(X_test, Y_test)

#loss = model.evaluate(X_test, Y_test, verbose=0)

#print("the loss is ",loss)

#s.heatmap(confusion_matrix(Y_test, model.predict(X_test)),annot=True)

model = Sequential()

# Configuring the parameters

model.add(Dense(n_hidden, input_shape=(timesteps, input_dim)))

model.add(Activation('exponential'))

# Adding a dropout layer

model.add(Dropout(0.1))

# Adding a dense output layer with sigmoid activation



#model.add(Dense(n_hidden, input_shape=(timesteps, input_dim),activation='sigmoid'))

model.add(Flatten())

model.add(Dense(6, activation='sigmoid'))

model.summary()



for i in p:

    model.compile(loss='categorical_crossentropy', optimizer=i)

    model.summary()

    print("Training...")

    model.fit(X_train, Y_train, nb_epoch=10, batch_size=16, validation_split=0.1,  verbose=2)

    loss = model.evaluate(X_test, Y_test, verbose=0)

    print("the loss is ",loss)

# we'll use categorical xent for the loss, and RMSprop as the optimizer

#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#model.summary()

#print("Training...")

#model.fit(X_train, Y_train, nb_epoch=10, batch_size=16, validation_split=0.1,  verbose=2)

#score = model.evaluate(X_test, Y_test)

#loss = model.evaluate(X_test, Y_test, verbose=0)

#print("the loss is ",loss)

#sns.heatmap(confusion_matrix(Y_test, model.predict(X_test)),annot=True)
model = Sequential()

# Configuring the parameters

model.add(Dense(n_hidden, input_shape=(timesteps, input_dim)))

model.add(Activation('selu'))

# Adding a dropout layer

model.add(Dropout(0.1))

# Adding a dense output layer with sigmoid activation



#model.add(Dense(n_hidden, input_shape=(timesteps, input_dim),activation='sigmoid'))

model.add(Flatten())

model.add(Dense(6, activation='sigmoid'))

model.summary()



# we'll use categorical xent for the loss, and RMSprop as the optimizer

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

model.summary()

print("Training...")

model.fit(X_train, Y_train, nb_epoch=10, batch_size=16, validation_split=0.1,  verbose=2)

score = model.evaluate(X_test, Y_test)

loss = model.evaluate(X_test, Y_test, verbose=0)

print("the loss is ",loss)

sns.heatmap(confusion_matrix(Y_test, model.predict(X_test)),annot=True)

#print(e)
model = Sequential()

# Configuring the parameters

model.add(Dense(n_hidden, input_shape=(timesteps, input_dim)))

model.add(Activation('elu'))

# Adding a dropout layer

model.add(Dropout(0.1))

# Adding a dense output layer with sigmoid activation



#model.add(Dense(n_hidden, input_shape=(timesteps, input_dim),activation='sigmoid'))

model.add(Flatten())

model.add(Dense(6, activation='sigmoid'))

model.summary()



# we'll use categorical xent for the loss, and RMSprop as the optimizer

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.summary()

print("Training...")

model.fit(X_train, Y_train, nb_epoch=10, batch_size=16, validation_split=0.1,  verbose=2)

score = model.evaluate(X_test, Y_test)

loss = model.evaluate(X_test, Y_test, verbose=0)

print("the loss is ",loss)

sns.heatmap(confusion_matrix(Y_test, model.predict(X_test)),annot=True)

#print(e)
model = Sequential()

# Configuring the parameters

model.add(Dense(n_hidden, input_shape=(timesteps, input_dim)))

model.add(Activation('softsign'))

# Adding a dropout layer

model.add(Dropout(0.1))

# Adding a dense output layer with sigmoid activation



#model.add(Dense(n_hidden, input_shape=(timesteps, input_dim),activation='sigmoid'))

model.add(Flatten())

model.add(Dense(6, activation='softmax'))

model.summary()



# we'll use categorical xent for the loss, and RMSprop as the optimizer

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

model.summary()

print("Training...")

model.fit(X_train, Y_train, nb_epoch=10, batch_size=16, validation_split=0.1,  verbose=2)

score = model.evaluate(X_test, Y_test)

loss = model.evaluate(X_test, Y_test, verbose=0)

print("the loss is ",loss)

sns.heatmap(confusion_matrix(Y_test, model.predict(X_test)),annot=True)

#print(e)
from keras.layers import LeakyReLU

model = Sequential()

# Configuring the parameters

model.add(Dense(n_hidden, input_shape=(timesteps, input_dim)))

model.add(LeakyReLU(alpha=0.1))

# Adding a dropout layer

model.add(Dropout(0.5))

# Adding a dense output layer with sigmoid activation



#model.add(Dense(n_hidden, input_shape=(timesteps, input_dim),activation='sigmoid'))

model.add(Flatten())

model.add(Dense(6, activation='softmax'))

model.summary()



# we'll use categorical xent for the loss, and RMSprop as the optimizer

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()

print("Training...")

model.fit(X_train, Y_train, nb_epoch=10, batch_size=16, validation_split=0.1,  verbose=2)

score = model.evaluate(X_test, Y_test)

loss = model.evaluate(X_test, Y_test, verbose=0)

print("the loss is ",loss)

sns.heatmap(confusion_matrix(Y_test, model.predict(X_test)),annot=True)

print(score)
from keras.layers import PReLU

model = Sequential()

# Configuring the parameters

model.add(Dense(n_hidden, input_shape=(timesteps, input_dim)))

model.add(PReLU())

# Adding a dropout layer

model.add(Dropout(0.1))

# Adding a dense output layer with sigmoid activation



#model.add(Dense(n_hidden, input_shape=(timesteps, input_dim),activation='sigmoid'))

model.add(Flatten())

model.add(Dense(6, activation='softmax'))

model.summary()



# we'll use categorical xent for the loss, and RMSprop as the optimizer

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

model.summary()

print("Training...")

model.fit(X_train, Y_train, nb_epoch=10, batch_size=16, validation_split=0.1,  verbose=2)

score = model.evaluate(X_test, Y_test)

loss = model.evaluate(X_test, Y_test, verbose=0)

print("the loss is ",loss)

sns.heatmap(confusion_matrix(Y_test, model.predict(X_test)),annot=True)

score
pd.DataFrame()

pd.add()
# Activities are the class labels

# It is a 6 class classification

ACTIVITIES = {

    0: 'WALKING',

    1: 'WALKING_UPSTAIRS',

    2: 'WALKING_DOWNSTAIRS',

    3: 'SITTING',

    4: 'STANDING',

    5: 'LAYING',

}



# Utility function to print the confusion matrix

def confusion_matrix(Y_true, Y_pred):

    Y_true = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_true, axis=1)])

    Y_pred = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_pred, axis=1)])



    return pd.crosstab(Y_true, Y_pred, rownames=['True'], colnames=['Pred'])

loss = model.evaluate(X_test, Y_test, verbose=0)

loss

e=confusion_matrix(Y_test, model.predict(X_test))
model = Sequential()

model.add(Dense(36, input_shape=(128,9)))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(6))

model.add(Flatten())



#model.add(Activation('softmax'))



# we'll use categorical xent for the loss, and RMSprop as the optimizer

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()

print("Training...")

model.fit(X_train, Y_train, nb_epoch=10, batch_size=16, validation_split=0.1,  verbose=2)

model.summary()
# Initiliazing the sequential model

model = Sequential()

# Configuring the parameters

model.add

model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim), ))

# Adding a dropout layer

model.add(Dropout(0.5))

# Adding a dense output layer with sigmoid activation

model.add(Dense(6, activation='sigmoid'))

model.summary()
# Compiling the model

model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
# Training the model

model.fit(X_train,

          Y_train,

          batch_size=batch_size,

          validation_data=(X_test, Y_test),

          epochs=epochs)
# Confusion Matrix

print(confusion_matrix(Y_test, model.predict(X_test)))
score = model.evaluate(X_test, Y_test)
score