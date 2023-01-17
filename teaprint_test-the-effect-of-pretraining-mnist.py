from sklearn.metrics import accuracy_score
from keras.datasets import mnist
import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard

from keras.layers import Input, Dense
from keras.models import Model
from keras.regularizers import l2
from keras.utils import to_categorical
import keras

import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

#### Load the data
file = open("../input/mnist_train.csv")
data_train = pd.read_csv(file)

y_train = np.array(data_train.iloc[:, 0])
x_train = np.array(data_train.iloc[:, 1:])

file = open("../input/mnist_test.csv")
data_test = pd.read_csv(file)
y_test = np.array(data_test.iloc[:, 0])
x_test = np.array(data_test.iloc[:, 1:])

x_train = x_train.astype('float32')/ 255.
x_test = x_test.astype('float32')/ 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_all = np.concatenate((x_train,x_test))
y_all = to_categorical(np.concatenate((y_train,y_test)))

print('Shape of x_train:',x_train.shape)
print('Shape of x_test:', x_test.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of y_test:',y_test.shape)


#### Split into train and test set
n_labeled = 1000
x_train = x_all[:n_labeled,:]
x_test = x_all[n_labeled:,:]
y_train = y_all[:n_labeled,:]
y_test = y_all[n_labeled:,:]

print(x_train.shape)
print(x_test.shape)
#### Construct neural Architeture for baseline model
input_img = Input(shape=(784,))
d = Dense(20, activation='relu')(input_img)
d = Dense(10, activation='relu')(d)
output = Dense(10, activation='softmax', kernel_regularizer=l2(0.01))(d)
baseline = Model(input_img,output)
baseline.compile(optimizer='adam', loss='binary_crossentropy',
                 metrics=['categorical_accuracy'])
#### Train the model
history_baseline = baseline.fit(x_train, y_train,
                epochs=1000,
                batch_size=100,
                shuffle=True,
                verbose=2,
                validation_data=[x_test,y_test])

#### Check the model performance
score = baseline.evaluate(x_test, y_test)
print ('keras test accuracy score:', score[1])

#### Check the model performance
score = baseline.evaluate(x_test, y_test)
print ('keras test accuracy score:', score[1])

#### Construct neural architecture of autoencoder
'''ref: https://towardsdatascience.com/unsupervised-learning-of-gaussian-\
mixture-models-on-a-selu-auto-encoder-not-another-mnist-11fceccc227e'''
# this is the size of our encoded representations
encoding_dim = 6  
# Specify the layer of autoencoder
input_img = Input(shape=(784,))
d = Dense(256, activation='selu')(input_img)
d = Dense(128, activation='selu')(d)
encoded = Dense(encoding_dim, activation='selu', kernel_regularizer=l2(0.01))(d)
d = Dense(128, activation='selu')(encoded)
d = Dense(256, activation='selu')(d)
decoded = Dense(784, activation='sigmoid')(d)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (6-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
deco = autoencoder.layers[-3](encoded_input)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
# create the decoder model
decoder = Model(encoded_input, deco)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#### Train the autoencoder. Note we use test data to train.
autoencoder.fit(x_test, x_test,
                epochs=500,
                batch_size=2000,
                shuffle=True,
                verbose=2,
                validation_data=(x_test, x_test))
#### Use encoder part of autoencoder to compress signal for supervised training
x_train_en = encoder.predict(x_train)
x_test_en = encoder.predict(x_test)
#### Construct neural architecture for autoencoder-pretrained model, same as baseline model.
input_encoded_img = Input(shape=(6,))
d = Dense(20, activation='relu')(input_encoded_img)
d = Dense(10, activation='relu')(d)
output = Dense(10, activation='softmax', kernel_regularizer=l2(0.01))(d)

classifier = Model(input_encoded_img,output)
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                 metrics=['categorical_accuracy'])
#### Train the model
history_pretrained = classifier.fit(x_train_en, y_train,
                epochs=1000,
                batch_size=100,
                shuffle=True,
                verbose=2,
                validation_data=[x_test_en,y_test])

#### Check the score 
score = classifier.evaluate(x_test_en, y_test)
print ('keras test accuracy score:', score[1])
#### Visualize train history
plt.plot(history_baseline.history['val_categorical_accuracy'])  
plt.plot(history_pretrained.history['val_categorical_accuracy'])  
plt.title('Train History')  
#plt.ylabel('')  
plt.xlabel('Epoch')  
plt.legend(['baseline:val_categorical_accuracy', 
            'pretrained:val_categorical_accuracy'], loc='lower right')  
plt.show()