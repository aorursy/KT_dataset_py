import tensorflow as tf

tf.__version__
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split # split train in train + validation

from tensorflow.keras import Sequential

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Lambda, Dense, Flatten, Dropout, BatchNormalization, Convolution2D , MaxPooling2D

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import RMSprop, Adam



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# notebook parameters

train_path = '/kaggle/input/digit-recognizer/train.csv'

test_path = '/kaggle/input/digit-recognizer/test.csv'
train = pd.read_csv(train_path)

test = pd.read_csv(test_path)

print('Dim of train: {}'.format(train.shape))

print('Dim of test: {}'.format(test.shape))
X_train = train.iloc[:, 1:].values.astype('float32')  # all pixel values

Y_train = train.iloc[:, 0].values.astype('int32')  # only labels i.e targets digits

X_test = test.values.astype('float32')
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print('Dim of train: {}'.format(X_train.shape))

print('Dim od test: {}'.format(X_test.shape))
j = 0

for i in range(6, 15):

    j += 1

    plt.subplot(330 + j)

    plt.imshow(X_train[i,:,:,:].reshape((28,28)), cmap=plt.get_cmap('gray'))

    plt.title(Y_train[i])
# ... feature standarization (later use)

mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)



def standardize(x):

    return (x-mean_px)/std_px
# ... one-hot-encoding of labels

Y_train = to_categorical(Y_train, num_classes=10)

num_classes = Y_train.shape[1]

print('Number of classes: {}'.format(num_classes))
# ... split the train data to train + validation (to monitor performance while training)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, 

                                                  test_size=0.10, random_state=777)
# ... define NN architecture

model = Sequential() # Initialize a Sequential object to define a NN layer by layer (sequentially)

model.add(Lambda(standardize, input_shape=(28,28,1))) # add a first layer which standardizes the input (grey-scale image of shape (28,28,1))

model.add(Flatten(data_format='channels_last')) # add a layer which transform the input of shape (28, 28, 1) to shape (n_x,)

model.add(Dense(10, activation='softmax')) # add a layer of 10 neurons that connect each one with all neurones in the previous layer

print("input shape ",model.input_shape)

print("output shape ",model.output_shape)
# ... define optimizer, loss function (and therefore a cost function) and metrics to monitor while training

model.compile(

    optimizer=RMSprop(lr=0.001, # learning rate (alpha)

                      rho=0.9, # momentum of order 2 (rho*dW + (1-rho)*dW**2)

                      momentum=0.0, # momentum of order 1 (rho*dW + (1-rho)*dW**2)

                      epsilon=1e-07, # term to avoid dividing by 0

                      centered=False, # if True standardize the gradients (high computational cost)

                      name='RMSprop'),

    loss='categorical_crossentropy',

    metrics=['accuracy'])
# ... fit NN (mini-batch approach of size 64 for 7 epochs with RMSprop optimizer)

history = model.fit(x=X_train,

                    y=Y_train,

                    batch_size=64,

                    epochs=20,

                    verbose=2,

                    validation_data=(X_val, Y_val))
# ... Visualize performance

history_dict = history.history

epochs = history.epoch

train_accuracy = history_dict['accuracy']

val_accuracy = history_dict['val_accuracy']



plt.figure()

plt.plot(epochs, train_accuracy, c='b')

plt.plot(epochs, val_accuracy, c='r')

plt.title('learning curves')

plt.ylim(0.9,1)

plt.ylabel('accuracy')

plt.xlabel('epochs')

plt.show()
# ... get real validation labels

real_val_class = []

for row in range(Y_val.shape[0]):

    prod = Y_val[row,:]*np.array(range(10))

    real_class = int(max(prod))

    real_val_class.append(int(real_class))
# ... validation error analysis

errors = pd.DataFrame({

    'real': real_val_class,

    'predict': model.predict_classes(X_val)

})



error_model_index = errors.loc[errors.real != errors.predict,:].index

error_model_index
i_check = error_model_index[0]

print('Real value: {}'.format(errors.loc[i_check,'real']))

print('Predict value: {}'.format(errors.loc[i_check,'predict']))



plt.figure(1)

plt.imshow(X_val[i_check,:,:,:].reshape((28,28)), cmap=plt.get_cmap('gray'))

plt.title(Y_val[i_check])
i_check = error_model_index[1]

print('Real value: {}'.format(errors.loc[i_check,'real']))

print('Predict value: {}'.format(errors.loc[i_check,'predict']))



plt.figure(1)

plt.imshow(X_val[i_check,:,:,:].reshape((28,28)), cmap=plt.get_cmap('gray'))

plt.title(Y_val[i_check])
# ... define NN architecture

model_complex = Sequential() # Initialize a Sequential object to define a NN layer by layer (sequentially)

model_complex.add(Lambda(standardize,input_shape=(28,28,1))) # add a first layer which standardizes the input (grey-scale image of shape (28,28,1))

model_complex.add(Flatten(data_format='channels_last')) # add a layer which transform the input of shape (28, 28, 1) to shape (n_x,)

model_complex.add(Dense(512, activation='relu')) # add a layer of 512 neurons that connect each one with all neurones in the previous layer

model_complex.add(Dense(256, activation='relu')) # add a layer of 512 neurons that connect each one with all neurones in the previous layer

model_complex.add(Dense(128, activation='relu')) # add a layer of 512 neurons that connect each one with all neurones in the previous layer

model_complex.add(Dense(64, activation='relu')) # add a layer of 512 neurons that connect each one with all neurones in the previous layer

model_complex.add(Dense(10, activation='softmax')) # add a layer of 10 neurons that connect each one with all neurones in the previous layer
model_complex.compile(optimizer='RMSprop', 

                      loss='categorical_crossentropy',

                      metrics=['accuracy'])
history_complex = model_complex.fit(x=X_train,

                                    y=Y_train,

                                    batch_size=64,

                                    epochs=20,

                                    verbose=2,

                                    validation_data=(X_val, Y_val))
# ... Visualize performance

history_dict = history_complex.history

epochs = history_complex.epoch

train_accuracy = history_dict['accuracy']

val_accuracy = history_dict['val_accuracy']



plt.figure()

plt.plot(epochs, train_accuracy, c='b')

plt.plot(epochs, val_accuracy, c='r')

plt.title('learning curves')

plt.ylim(0.9,1)

plt.ylabel('accuracy')

plt.xlabel('epochs')

plt.show()
# ... predictions

predictions = model_complex.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                          "Label": predictions})

submissions.to_csv("model_complex.csv", index=False, header=True)
# ... define architecture

cnn_model = Sequential()

cnn_model.add(Lambda(standardize, input_shape=(28,28,1)))

cnn_model.add(Convolution2D(64,(3,3), activation='relu'))

cnn_model.add(BatchNormalization())

cnn_model.add(MaxPooling2D())

cnn_model.add(Flatten())

cnn_model.add(Dense(124, activation='relu'))

cnn_model.add(BatchNormalization())

cnn_model.add(Dropout(0.9))

cnn_model.add(Dense(10, activation='softmax'))
# ... define optimizer, loss function and metrics to monitor

cnn_model.compile(optimizer=Adam(learning_rate=0.0001), 

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])
history_cnn = cnn_model.fit(x=X_train,

                            y=Y_train,

                            batch_size=256,

                            epochs=200,

                            verbose=2,

                            validation_data=(X_val, Y_val))
# ... Visualize performance

history_dict = history_cnn.history

epochs = history_cnn.epoch

train_accuracy = history_dict['accuracy']

val_accuracy = history_dict['val_accuracy']



plt.figure()

plt.plot(epochs, train_accuracy, c='b')

plt.plot(epochs, val_accuracy, c='r')

plt.title('learning curves')

plt.ylim(0.9,1)

plt.ylabel('accuracy')

plt.xlabel('epochs')

plt.show()
predictions = cnn_model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                          "Label": predictions})

submissions.to_csv("cnn_simple.csv", index=False, header=True)
# ... define architecture

cnn_adam_model = Sequential()

cnn_adam_model.add(Lambda(standardize, input_shape=(28,28,1)))

cnn_adam_model.add(Convolution2D(1024,(3,3), activation='relu'))

cnn_adam_model.add(BatchNormalization())

cnn_adam_model.add(MaxPooling2D())

cnn_adam_model.add(Convolution2D(512,(3,3), activation='relu'))

cnn_adam_model.add(BatchNormalization())

cnn_adam_model.add(MaxPooling2D())

cnn_adam_model.add(Convolution2D(256,(3,3), activation='relu'))

cnn_adam_model.add(BatchNormalization())

cnn_adam_model.add(MaxPooling2D())

cnn_adam_model.add(Flatten())

cnn_adam_model.add(Dense(512, activation='relu'))

cnn_adam_model.add(BatchNormalization())

cnn_adam_model.add(Dropout(0.9))

cnn_adam_model.add(Dense(124, activation='relu'))

cnn_adam_model.add(BatchNormalization())

cnn_adam_model.add(Dropout(0.9))

cnn_adam_model.add(Dense(10, activation='softmax'))
# ... define optimizer, loss function and metrics to monitor

cnn_adam_model.compile(optimizer=Adam(), 

                       loss='categorical_crossentropy',

                       metrics=['accuracy', 'mse'])
history_cnn_adam = cnn_adam_model.fit(x=X_train,

                                      y=Y_train,

                                      batch_size=124,

                                      epochs=50,

                                      verbose=2,

                                      validation_data=(X_val, Y_val))
# ... Visualize performance

history_dict = history_cnn_adam.history

epochs = history_cnn_adam.epoch

train_accuracy = history_dict['accuracy']

val_accuracy = history_dict['val_accuracy']



plt.figure()

plt.plot(epochs, train_accuracy, c='b')

plt.plot(epochs, val_accuracy, c='r')

plt.title('learning curves')

plt.ylim(0.9,1)

plt.ylabel('accuracy')

plt.xlabel('epochs')

plt.show()
predictions = cnn_adam_model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                          "Label": predictions})

submissions.to_csv("cnn_complex.csv", index=False, header=True)