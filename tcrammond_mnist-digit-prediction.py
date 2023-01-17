# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from random import randint
import matplotlib.pyplot as plt
import seaborn as sns

# Importing Keras layers
import keras
from keras.models import Sequential # the model we use to hold our CNN layers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#import the MNIST data
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.sample(5)
test.sample(5)
# /get the number of features (n), and the number of examples (m) from the test and train data
(m_train, n_train) = train.shape
(m_test, n_test) = test.shape
# Split the training set between the labels and the training data
train_labels = train['label']
train = train.drop(['label'], axis=1)
# Convert the training data to a 3D array for visualization purposes
train_images = train.to_numpy().reshape(42000, 28, 28)
test_images = test.to_numpy().reshape(28000, 28, 28)
# Plot 4 random images from the training data set, with the corresponding labels
for i in range(0, 4):
    train_example = np.random.randint(m_train)
    plt.subplot(220 + (i+1))
    plt.title(train_labels[train_example])
    plt.imshow(train_images[train_example], cmap='gray')
    plt.grid(False)
    plt.axis('off')

# Plot 4 random images from the test data set, with the corresponding labels
for i in range(0, 4):
    test_example = np.random.randint(m_test)
    plt.subplot(220 + (i+1))
    plt.imshow(test_images[test_example], cmap='gray')
    plt.grid(False)
    plt.axis('off')

sns.countplot(train_labels)
# create a keras sequential model
model = keras.Sequential()

# add convolution layers.
model.add(Conv2D(32, kernel_size=(3,3),
                activation='relu',
                input_shape=(28,28,1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
          
#view a summary of the model's different layers
model.summary()
# compile the model
model.compile(optimizer='adam',
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
# Reshape the data into a format that can be used by the CNN
X_train = train.to_numpy().reshape(42000, 28, 28,1)
X_test = test.to_numpy().reshape(28000, 28, 28,1)
# Convert the labels into a binary class matrix (also known as one-hot encoding)
y_train = keras.utils.to_categorical(train_labels, 10)
# training the Model. Initializing the 'model_history' variable so we can retrieve some information about how the model performed
model_history = model.fit(X_train, y_train,
                          batch_size=32,
                          epochs=10,
                          verbose=1,
                          validation_split=0.1)
print(model_history.history.keys())
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Using the model to generate predicitons 
test_predictions = model.predict(X_test)
# plot a random image from the test data set compared against the prediction_results output

# convert the prediciton probabilities to the actual labels
labels = np.argmax(test_predictions, axis=1)

# plot 4 random test images, with the generated prediction label
for i in range(0, 4):
    index = np.random.randint(m_test)
    plt.subplot(220 + (i+1))
    plt.title(' Prediction: ' + str(labels[index]))
    plt.imshow(test_images[index], cmap='gray')
    plt.grid(False)
    plt.axis('off')



submission = pd.DataFrame({'ImageID': list(range(1, len(test_predictions)+1)), 'Label': labels})
submission.to_csv('MNIST_Digit_Predictions.csv', index=False, header=True)