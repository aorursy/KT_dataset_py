import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import keras

from keras.models import Model

from keras.layers import *

from keras import optimizers



from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input"))



from matplotlib import pyplot



np.random.seed(1212)

batch_size = 128

num_classes = 10

epochs = 10
# mnist data

df_train = pd.read_csv('../input/mnist_train.csv')

df_test = pd.read_csv('../input/mnist_test.csv')

print(df_train.shape)



#Validation split from training data

df_features = df_train.iloc[:, 1:785]

df_label = df_train.iloc[:, 0]

X_train, X_cv, y_train, y_cv = train_test_split(df_features, df_label, test_size = 0.2, random_state = 1212)



#Normalization : Very Important

X_train = X_train.values.astype('float32')/255.

X_cv = X_cv.values.astype('float32')/255.



X_test = df_test.iloc[:, 1:785]

y_test = df_test.iloc[:, 0]

X_test = X_test.values.astype('float32')/255.



# Convert labels to One Hot Encoded

y_train = keras.utils.to_categorical(y_train, num_classes)

y_cv = keras.utils.to_categorical(y_cv, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)



print(X_train.shape, X_cv.shape, X_test.shape)
# model : ADAM + 4 hidden + alpha=0.1 + dropout 

# Input Parameters

n_input = 784 # number of features

n_hidden_1 = 300

n_hidden_2 = 100

n_hidden_3 = 100

n_hidden_4 = 200

num_digits = 10



# Insert Hyperparameters

learning_rate = 0.1

training_epochs = 20

batch_size = 100

dropout_factor = 0.3



Inp = Input(shape=(784,))

x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)

x = Dropout(dropout_factor)(x)

x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)

x = Dropout(dropout_factor)(x)

x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)

x = Dropout(dropout_factor)(x)

x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)



model = Model(Inp, output)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size = batch_size, verbose=2, epochs = training_epochs, validation_data=(X_cv, y_cv))
# plot loss during training

# plot accuracy during training

pyplot.subplot(212)

pyplot.title('Accuracy')

pyplot.plot(history.history['acc'], label='train') 

pyplot.plot(history.history['val_acc'], label='val') 

pyplot.legend()

pyplot.show()
score, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(score, test_acc)
import random

image_idx = random.randint(1,10000)-1

first_image = X_test[image_idx]

pixels = first_image.reshape((28, 28))

pyplot.imshow(pixels, cmap='gray')

pyplot.show()

predictions = model.predict(X_test, batch_size=200)

print(predictions[image_idx].argmax(axis=0))