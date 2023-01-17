# importing the essential libraries

import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

# importing the libraries required for neural networks

from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
# getting the training data

training_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
training_data.head()
# splitting the input features and target features.

X = training_data.drop(['label'], axis = 1)

y = training_data['label'].values
X.head()
# Tensorflow layers expects the inputs in the form of Array 

# Conv2D expects the dimension to be ndim=4

X = np.array(X)

X = X.reshape(42000, 28, 28, 1)
# Splitting the data into training and evaluation

# holdout validation method

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
classes = len(set(y))
# I am creating a neural network with 13 deep hidden layers + 1 input and 1 output layer.

cnn_model = Sequential(name = 'C-Neural_Network_Model')

cnn_model.add(Conv2D(filters = 64, kernel_size = (5, 5), activation = 'relu', input_shape = (28, 28, 1), name = 'First_layer'))

cnn_model.add(MaxPooling2D(pool_size = (2, 2), name = 'Second_layer'))

cnn_model.add(Dropout(rate = 0.2, name = 'Third_layer'))

cnn_model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', name = 'Fourth_layer'))

cnn_model.add(MaxPooling2D(pool_size = (2, 2), name = 'Fifth_layer'))

cnn_model.add(Dropout(rate = 0.2, name = 'Sixth_layer'))

cnn_model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', name = 'Seventh_layer'))

cnn_model.add(MaxPooling2D(pool_size = (2, 2), name = 'Eighth_layer'))

cnn_model.add(Dropout(rate = 0.15, name = 'nineth_layer'))

cnn_model.add(Flatten(name = 'tenth_layer'))

cnn_model.add(Dense(units = 128, activation = 'relu', name = 'eleventh_layer'))

cnn_model.add(Dropout(rate = 0.1, name = 'twelfth_layer'))

cnn_model.add(Dense(units = 32, activation = 'relu', name = 'Thirteenth_layer'))

cnn_model.add(Dropout(rate = 0.1, name = 'fourteenth_layer'))

cnn_model.add(Dense(units = classes, activation = 'softmax', name = 'Output_layer'))

cnn_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
fitted_model = cnn_model.fit(X_train, y_train, epochs = 25)
# plotting the accuracy for each epoch

accuracy = fitted_model.history['acc']

plt.plot(range(len(accuracy)), accuracy, 'o', label = 'accuracy')

plt.title('Accuracy of the model for each epoch')

plt.legend()
evaluation = cnn_model.predict_classes(X_test)
print(classification_report(evaluation, y_test))
testing_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
testing_data.shape
# doing the same preprocessing steps that we done in training data

testing_data = np.array(testing_data)

testing_data = testing_data.reshape(28000, 28, 28, 1)
predictions = cnn_model.predict_classes(testing_data)
submissions = pd.DataFrame({'ImageId':range(1, len(testing_data)+1), 'Label':predictions})
submissions.head()
submissions.to_csv('submissions.csv', index = False)