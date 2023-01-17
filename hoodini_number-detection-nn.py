import pandas as pd

import numpy as np



np.random.seed(1212)



import keras

from keras.models import Model

from keras.layers import *

from keras import optimizers
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_features = df_train.iloc[:, 1:785]

df_label = df_train.iloc[:, 0]



X_test = df_test.iloc[:, 0:784]



print(X_test.shape)
from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(df_features, df_label, 

                                                test_size = 0.2,

                                                random_state = 1212)

len(X_train)
# Feature Normalization 

X_train = X_train.astype('float32'); X_cv= X_cv.astype('float32'); X_test = X_test.astype('float32')

X_train /= 255; X_cv /= 255; X_test /= 255



# Convert labels to One Hot Encoded

num_digits = 10

y_train = keras.utils.to_categorical(y_train, num_digits)

y_cv = keras.utils.to_categorical(y_cv, num_digits)
# Printing 2 examples of labels after conversion

print(y_train[0]) # 2

print(y_train[3]) # 7
n_input = 784 # number of features

n_hidden_1 = 300

n_hidden_2 = 100

n_hidden_3 = 100

n_hidden_4 = 200

num_digits = 10
Inp = Input(shape=(784,))

x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)

x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)

x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)

x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)
model = Model(Inp, output)

model.summary()
learning_rate = 0.1

training_epochs = 20

batch_size = 100

sgd = optimizers.SGD(lr=learning_rate)
# We rely on the plain vanilla Stochastic Gradient Descent as our optimizing methodology

model.compile(loss='categorical_crossentropy',

              optimizer='sgd',

              metrics=['accuracy'])
history1 = model.fit(X_train, y_train,

                     batch_size = batch_size,

                     epochs = training_epochs,

                     verbose = 2,

                     validation_data=(X_cv, y_cv))
Inp = Input(shape=(784,))

x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)

x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)

x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)

x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)



# We rely on ADAM as our optimizing methodology

adam = keras.optimizers.Adam(lr=learning_rate)

model2 = Model(Inp, output)



model2.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history2 = model2.fit(X_train, y_train,

                      batch_size = batch_size,

                      epochs = training_epochs,

                      verbose = 2,

                      validation_data=(X_cv, y_cv))
Inp = Input(shape=(784,))

x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)

x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)

x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)

x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)



learning_rate = 0.01

adam = keras.optimizers.Adam(lr=learning_rate)

model2a = Model(Inp, output)



model2a.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history2a = model2a.fit(X_train, y_train,

                        batch_size = batch_size,

                        epochs = training_epochs,

                        verbose = 2,

                        validation_data=(X_cv, y_cv))
Inp = Input(shape=(784,))

x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)

x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)

x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)

x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)



learning_rate = 0.5

adam = keras.optimizers.Adam(lr=learning_rate)

model2b = Model(Inp, output)



model2b.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history2b = model2b.fit(X_train, y_train,

                        batch_size = batch_size,

                        epochs = training_epochs,

                            validation_data=(X_cv, y_cv))
# Input Parameters

n_input = 784 # number of features

n_hidden_1 = 300

n_hidden_2 = 100

n_hidden_3 = 100

n_hidden_4 = 200

num_digits = 10
Inp = Input(shape=(784,))

x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)

x = Dropout(0.3)(x)

x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)

x = Dropout(0.3)(x)

x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)

x = Dropout(0.3)(x)

x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)
# Our model would have '6' layers - input layer, 4 hidden layer and 1 output layer

model4 = Model(Inp, output)

model4.summary() 
model4.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history = model4.fit(X_train, y_train,

                    batch_size = batch_size,

                    epochs = training_epochs,

                    validation_data=(X_cv, y_cv))
test_pred = pd.DataFrame(model4.predict(X_test, batch_size=200))

test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))

test_pred.index.name = 'ImageId'

test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()

test_pred['ImageId'] = test_pred['ImageId'] + 1



test_pred.head()
test_pred.to_csv('mnist_submission.csv', index = False)