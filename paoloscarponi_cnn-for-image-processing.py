# modules import

import keras

import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
# data import and analysis

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

print('DATASET BASIC STRUCTURE')
print('')
print('Training data shape : ', X_train.shape, Y_train.shape)
print('Testing data shape : ', X_test.shape, Y_test.shape)
# fundamental information extraction

number_of_training_records = X_train.shape[0]
number_of_testing_records = X_test.shape[0]

record_dimension = X_train.shape[1:]
label_dimension = Y_train.shape[1:]

classes = np.unique(Y_train)
number_of_classes = len(classes)

print('DATASET BASIC INFORMATION')
print('')
print('Number of training records:', number_of_training_records)
print('Number of testing records:', number_of_testing_records)
print('')
print('Record dimension:', record_dimension)
print('')
print('Total number of classes: ', number_of_classes)
print('Output classes: ', classes)
# data preprocessing - a-priori information printing
print('POST-PROCESSING DATASET INFORMATION')
print('')
print('Training data type:', type(X_train[0, 0, 0]))
print('Testing data type:', type(X_test[0, 0, 0]))
print('')
print('Training data definition interval: [' + str(X_train.min()) + ', ' + str(X_train.max()) + ']')
print('Testing data definition interval: [' + str(X_test.min()) + ', ' + str(X_test.max()) + ']')
print('')

# data preprocessing - conversion to float

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# data preprocessing - normalization

X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

# data preprocessing - reshape

X_train = X_train.reshape(number_of_training_records, record_dimension[0], record_dimension[1], 1)
X_test = X_test.reshape(number_of_testing_records, record_dimension[0], record_dimension[1], 1)

# data preprocessing - a-posteriori information printing
print('POST-PROCESSING DATASET INFORMATION')
print('')
print('Training data type:', type(X_train[0, 0, 0, 0]))
print('Testing data type:', type(X_test[0, 0, 0, 0]))
print('')
print('Training data definition interval: [' + str(X_train.min()) + ', ' + str(X_train.max()) + ']')
print('Testing data definition interval: [' + str(X_test.min()) + ', ' + str(X_test.max()) + ']')
# data preprocessing - labels one-hot encoding
Y_train_one_hot = to_categorical(Y_train)
Y_test_one_hot = to_categorical(Y_test)

# data preprocessing - training/validation data split
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train_one_hot, test_size=0.2, random_state=0)
# neural network construction

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(28, 28, 1)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2), padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))           
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(number_of_classes, activation='softmax'))
# model compilation

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

# model summary visualization

fashion_model.summary()
# model training

fashion_train = fashion_model.fit(X_train, Y_train, batch_size=64, epochs=20, verbose=1, validation_data=(X_validation, Y_validation))

# model testing

test_eval = fashion_model.evaluate(X_test, Y_test_one_hot, verbose=0)

# performances evaluation

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
# performances analysis

accuracy = fashion_train.history['accuracy']
val_accuracy = fashion_train.history['val_accuracy']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# labels prediciton on test dataset

predicted_classes = fashion_model.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

# printing classification report on test data

target_names = ["Class {}".format(i) for i in range(number_of_classes)]
print(classification_report(Y_test, predicted_classes, target_names=target_names))