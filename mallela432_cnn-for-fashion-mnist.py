# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Add
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from tensorflow.python import keras

img_rows, img_cols = 28, 28
num_classes = 10
batch_size = 128
num_classes = 10
epochs = 25

def prep_data(raw):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y
# preparing the data for Training
fashion_train_data = "../input/fashion-mnist_train.csv"
fashion_train = np.loadtxt(fashion_train_data, skiprows=1, delimiter=',')
X_train, Y_train = prep_data(fashion_train)

# preparing the data for Testing
fashion_test_data = "../input/fashion-mnist_test.csv"
fashion_test = np.loadtxt(fashion_test_data, skiprows=1, delimiter=',')
x_test, y_test = prep_data(fashion_test)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout,MaxPooling2D

fashion_model = Sequential() 
fashion_model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(img_rows,img_cols,1)))
fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
fashion_model.add(MaxPooling2D(pool_size=(2, 2)) )
fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
fashion_model.add(Dropout(0.2))
fashion_model.add(Conv2D(10, 1, activation='relu'))
fashion_model.add(Conv2D(10, 9))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='relu'))
fashion_model.add(Dense(num_classes, activation='softmax'))
fashion_model.summary()
fashion_model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
visualize = fashion_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,validation_split = 0.2, verbose=1)
score = fashion_model.evaluate(x_test, y_test, verbose=0)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
import matplotlib.pyplot as plt
%matplotlib inline
accuracy = visualize.history['acc']
val_accuracy = visualize.history['val_acc']
loss = visualize.history['loss']
val_loss = visualize.history['val_loss']
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
#get the predictions for the test data
predicted_classes = fashion_model.predict_classes(x_test)

#get the indices to be plotted
y_true = fashion_test[:, 0]
correct = np.nonzero(predicted_classes == y_true)[0]
incorrect = np.nonzero(predicted_classes != y_true)[0]

# adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7,14)

figure_evaluation = plt.figure()

# plot 6 correct predictions
for i, correct in enumerate(correct[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted: {}, Truth: {}".format(predicted_classes[correct],
                                        y_true[correct]))
    plt.xticks([])
    plt.yticks([])


# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect[:6]):
    plt.subplot(6,3,i+10)
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted {}, Truth: {}".format(predicted_classes[incorrect], 
                                       y_true[incorrect]))
    plt.xticks([])
    plt.yticks([])


from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names=target_names))