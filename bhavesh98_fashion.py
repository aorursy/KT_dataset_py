#Importing libs

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import random

import seaborn as sns

from sklearn.model_selection import train_test_split

import keras

from keras import Sequential

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
#Importing dataset

fashion_train_df = pd.read_csv('../input/fashion-mnist_train.csv', sep = ',')

fashion_test_df = pd.read_csv('../input/fashion-mnist_test.csv', sep = ',')
#Visualizing the data

fashion_train_df.head()
fashion_test_df.head()
fashion_test_df.tail()
fashion_test_df.shape
fashion_train_df.shape
#Making arrays of the dataframes for visualizing the images

training_array = np.array(fashion_train_df, dtype = 'float32')

testing_array = np.array(fashion_test_df, dtype = 'float32')
i = random.randint(1, 60000)

plt.imshow(training_array[i, 1:].reshape(28, 28))

label = training_array[i, 0]

label
W_grid = 15

L_grid = 15

fig, axes = plt.subplots(L_grid, W_grid, figsize = (20, 20))

axes = axes.ravel()

n_training = len(training_array)

for i in np.arange(0, W_grid * L_grid):

    index = np.random.randint(0, n_training)

    axes[i].imshow(training_array[index, 1:].reshape((28, 28)))

    axes[i].set_title(training_array[index, 0], fontsize = 8)

    axes[i] .axis('off')

    

plt.subplots_adjust(hspace = 0.5)
#Making the train and test set

X_train = training_array[:, 1:]/255

y_train = training_array[:, 0]

X_test = training_array[:, 1:]/255

y_test = training_array[:, 0]
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))

X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))

X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))
#Starting the CNN

cnn_model = Sequential()
#Adding convolution

cnn_model.add(Convolution2D(64, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))
#Adding the max pooling

cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
#Flattening 

cnn_model.add(Flatten())
#Adding the Artificial neural network

#Adding hidden layer

cnn_model.add(Dense(output_dim = 64, activation = 'relu'))
#Adding the output layer

cnn_model.add(Dense(output_dim = 10, activation = 'sigmoid'))
cnn_model.compile(loss = 'sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'] )
epochs = 50

cnn_model.fit(X_train, y_train, batch_size = 512, epochs = epochs, validation_data=(X_validate, y_validate))
#Evaluation

evaluation = cnn_model.evaluate(X_test, y_test)

print('Test accuracy = {:.3f}'.format(evaluation[1]))
predicted_classes = cnn_model.predict_classes(X_test)
#Plotting confusion matrix

cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize = (14, 10))

sns.heatmap(cm, annot = True)
#Viewing the classification report

num_classes = 10

target_names = ['Class {}'.format(i) for i in range(num_classes)]

print(classification_report(y_test, predicted_classes, target_names = target_names))