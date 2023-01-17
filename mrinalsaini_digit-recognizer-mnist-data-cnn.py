# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# linear algebra

import numpy as np



# data processing

import pandas as pd



# data visualization

import matplotlib.pyplot as plt

import seaborn as sns



# neural network

import tensorflow as tf



# data preprocessing

from sklearn.model_selection import train_test_split
# Importing the training and testing datasets

training = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

testing = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
training.shape
training.head()
# Number of instances of each number

plt.figure(figsize = (12,6))



graph = sns.countplot(training['label'], palette = 'OrRd')

for p in graph.patches:

  graph.annotate(s = p.get_height(), 

                 xy = (p.get_x() + p.get_width()/2, p.get_height()),

                 xytext = (0,5),

                 textcoords="offset points",

                 ha = 'center',

                 va = 'center')
# Viewing randomly selected numbers and their labels

width = 15

length = 15



plt.figure(figsize=(20,30))

for i in range(length * width):

  plt.subplot(length, width, i+1)

  index = np.random.randint(0, len(training))

  plt.gca().set_title(training.iloc[index, 0], fontsize = 15)

  plt.imshow(training.iloc[index, 1:].values.reshape(28, 28))

  plt.axis('off')
# Separating the label and the pixel matrix

X = training.drop(columns = 'label').values

y = training['label'].values
# Rescaling and reshaping the pixel matrix

X = X/255

X = X.reshape(X.shape[0], *(28, 28, 1))

print(X.shape)
# Rescaling and reshaping the testing data

testing = testing.values

testing = testing/255

testing = testing.reshape(testing.shape[0], *(28, 28, 1))
# Splitting the training data into validation and training sets

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)
cnn = tf.keras.models.Sequential()
cnn.add(layer = tf.keras.layers.Conv2D(input_shape = (28, 28, 1),

                                       filters=64,

                                       kernel_size = (3,3), 

                                       activation = 'relu'))
cnn.add(layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides = 2))
cnn.add(layer = tf.keras.layers.Conv2D(filters=64,

                                       kernel_size = (3,3), 

                                       activation = 'relu'))
cnn.add(layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides = 2))
cnn.add(layer = tf.keras.layers.Flatten())
cnn.add(layer = tf.keras.layers.Dropout(0.5))
cnn.add(layer = tf.keras.layers.Dense(units = 108, activation='relu'))
cnn.add(layer = tf.keras.layers.Dropout(0.5))
cnn.add(layer = tf.keras.layers.Dense(units = 108, activation='relu'))
cnn.add(layer = tf.keras.layers.Dense(units = 10, activation = 'softmax'))
cnn.compile(optimizer = 'adam',

             loss = 'sparse_categorical_crossentropy',

             metrics = ['accuracy'])
model_train = cnn.fit(x = X_train, y = y_train, validation_data = (X_val, y_val), epochs = 25)
# Observing the changes in the validation and training loss

plt.plot(list(range(1, 26)), model_train.history['loss'], 'y', label='Training Loss')

plt.plot(list(range(1, 26)), model_train.history['val_loss'], 'r', label='Validation Loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# Observing the changes in the validation and training accuracy

plt.plot(list(range(1, 26)), model_train.history['accuracy'], 'y', label='Training Accuracy')

plt.plot(list(range(1, 26)), model_train.history['val_accuracy'], 'r', label='Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
# Predicting the labels of the testing set

y_pred = np.argmax(cnn.predict(testing), axis = -1)
y_pred
predictions = pd.DataFrame({"ImageId" : list(range(1, len(y_pred)+1)),

                             "Label" : y_pred})
# Saving the predicted results in a csv file

predictions.to_csv('predictions.csv', index = False)