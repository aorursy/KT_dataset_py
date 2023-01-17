# importing the essential libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# importing the deep learning libraries

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# getting the training data from the location

train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
train_data.head()
# Checking the distribution of the labels in the dataset

train_data['label'].value_counts()
# Splitting the training and validating data

X = train_data.drop(['label'], axis = 1).values

y = train_data[['label']].values



# scaling the data

X = X / 255.0



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 101)
print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)
# Reshaping the data

X_train = X_train.reshape(48000, 28, 28, 1)

X_val = X_val.reshape(12000, 28, 28, 1)
print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)
plt.imshow(X_train[0][:,:,0])
# the total number of outputs

output_classes = len(set(train_data['label']))
# preparing the artificial neural network with CNN Layers

cnn_model = Sequential()

cnn_model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))

cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

# cnn_model.add(Dropout(rate = 0.2))

cnn_model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))

cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

cnn_model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))

cnn_model.add(Flatten())

cnn_model.add(Dense(units = 128, activation = 'relu'))

# cnn_model.add(Dropout(rate = 0.2))

cnn_model.add(Dense(units = 64, activation = 'relu'))

# cnn_model.add(Dropout(rate = 0.1))

cnn_model.add(Dense(units = output_classes, activation = 'softmax'))



cnn_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
fitted_model = cnn_model.fit(X_train, y_train, epochs = 10, validation_data = (X_val, y_val))
accuracy = fitted_model.history['acc']

x_axis = range(len(accuracy))

plt.plot(x_axis, accuracy, 'o', label = 'accuracy plot')

plt.legend()
test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

test_data.head()
X = test_data.drop(['id'], axis = 1)

X = X / 255.0

print(X.shape)
X = X.values.reshape(5000, 28, 28, 1)
predictions = cnn_model.predict_classes(X)
# submission = pd.DataFrame({'id' : test_data['id'], 'label' : predictions})

# submission.head()
submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predictions

submission.head()
submission.to_csv("submission.csv",index=False)