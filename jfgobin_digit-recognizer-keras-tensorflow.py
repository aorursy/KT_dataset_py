# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
K.set_image_dim_ordering('th')

# Set the random seed for reproducibility
seed = 37
np.random.seed(seed)

# Read the datasets
csv_train = pd.read_csv("../input/train.csv")
csv_test = pd.read_csv("../input/test.csv")

# Separate the data from the results in the training set
y_train = csv_train.iloc[:,0]
X_train = csv_train.iloc[:,1:]

# Reshape the "image" part
y_train = y_train.as_matrix()
X_train = X_train.as_matrix().reshape(X_train.count()[0], 1, 28, 28).astype(np.float32)
X_test = csv_test.as_matrix().reshape(csv_test.count()[0], 1, 28, 28).astype(np.float32)

# Normalize the input
X_mean = X_train.mean().astype(np.float32)
X_sdev = X_train.std().astype(np.float32)

X_train = (X_train - X_mean)/X_sdev
X_test = (X_test - X_mean)/X_sdev

# Convert the labels 0...9 to categoricals
y_train = np_utils.to_categorical(y_train)

# Create the model
num_classes=10
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu',
          bias_initializer='RandomNormal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(23, activation='relu'))
#model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile the model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Partition the train set into training and validation
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                  test_size=0.1, random_state=83)
# Fit the model
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val), epochs=10,
                    batch_size=100, verbose=1)
history_dict = history.history
# Plot the loss and accuracy values
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, len(loss_values) + 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss_values, 'bo')
# b+ is for "blue crosses"
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.show()
# And let's predict the labels
y_pred = model.predict(X_test, batch_size=2000, 
                       verbose=1)

# Move from categorical back to 0...9
y_pred = np.argmax(y_pred, axis=1)
to_submit = pd.DataFrame({'Label': y_pred})
# Re-index to start at 1 instead of 0
to_submit.index += 1
to_submit.index.name = "ImageId"
to_submit.to_csv('labelled.csv')
