# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split



# seed for reproducing same results

seed = 5

np.random.seed(seed)



# read input and create features (X) and labels (y) variables

data_train = pd.read_csv('../input/train.csv')

X = data_train.drop('label', axis=1)

y = data_train['label']



# create the train-test-split

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1, random_state=seed)



print('There are %d train entries and %d test entries' % (len(X_train), len(y_test)))
%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm



# transform our pandas dataframes to np-arrays

X_train_matrix = X_train.as_matrix()

y_train_matrix = y_train.as_matrix()

X_test_matrix = X_test.as_matrix()

y_test_matrix = y_test.as_matrix()



# plot first ten training images

fig = plt.figure(figsize=(20,20))

for i in range(10):

    axis = fig.add_subplot(1, 10, i+1, xticks=[], yticks=[])

    # reshape the image-data from a 1D array to a 2D array with shape (28,28)

    # as the mnist images are squares of 784 pixels

    axis.imshow(np.reshape(X_train_matrix[i], (28,28)), cmap='gray')

    axis.set_title(str(y_train_matrix[i]))
# rescale the image data to be between 0 and 1

print('before:')

print(X_train_matrix[0])



X_train_matrix = X_train_matrix.astype('float32') / 255

X_test_matrix = X_test_matrix.astype('float32') / 255



print('after:')

print(X_train_matrix[0])
# one-hot-encode the labels

from keras.utils import np_utils



print('Before:')

print(y_train_matrix[:8])



# we have 10 categories: 0 to 9

y_train_matrix = np_utils.to_categorical(y_train_matrix, 10)

y_test_matrix = np_utils.to_categorical(y_test_matrix, 10)



print('After:')

print(y_train_matrix[:8])
from keras.models import Sequential

from keras.layers import Dense, Dropout



# let's start with a basic Multilayer Perceptron (MLP) model architecture

# we will compare this to the CNN model later

model = Sequential()

model.add(Dense(784, input_shape=(len(X_train_matrix[0]),)))

model.add(Dense(392, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(196, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



# summarize the model's architecture

model.summary()
# finally, compile the model

# the accuracy metric will allow us to see how the accuracy changes during training

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# the model starts with random weights

# let's check if this shows when eveluating test accuracy

score = model.evaluate(X_test_matrix, y_test_matrix, verbose=0)

accuracy = score[1] * 100

# we'd expect this to be around 10%, as there are 10 different categories (numbers 0-9)

print('Random test accuracy is %1.1f%%' % accuracy)
# let's train the model!

history = model.fit(X_train_matrix, y_train_matrix, batch_size=100, epochs=6, verbose=1)
# let's try some more advanced fitting, including validation & shuffle

# also save the model with the best accuracy

from keras.callbacks import ModelCheckpoint



cb_checkpoint = ModelCheckpoint(filepath='best-model.hdf5', verbose=1, save_best_only=True)



history = model.fit(X_train_matrix, y_train_matrix, batch_size=400, epochs=6, 

                    validation_split=0.2, shuffle=True, callbacks=[cb_checkpoint], verbose=1)
# using a validation set during training did improve accuracy

# it also shows us whether the model tends to overfit our training data



# now, let's load the model with best validation accuracy

model.load_weights('best-model.hdf5')



# let's test our model on the test set

score = model.evaluate(X_test_matrix, y_test_matrix, verbose=0)

accuracy = score[1] * 100

print("Best Model's test accuracy is %1.1f%%" % accuracy)
# transform our data to be used in CNNs

X_train = np.array(list(map(lambda x: np.reshape(x, (28,28,1)), X_train_matrix)))

X_test = np.array(list(map(lambda x: np.reshape(x, (28,28,1)), X_test_matrix)))
# Now we define our CNN architecture

# I chose to pretty much use the one defined in TensorFlow's 

# Deep MNIST for Experts Tutorial (https://www.tensorflow.org/get_started/mnist/pros)



from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', 

                 input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))



# summarize the model's architecture

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# the model starts with random weights

# let's check if this shows when eveluating test accuracy

score = model.evaluate(X_test, y_test_matrix, verbose=0)

accuracy = score[1] * 100

# we'd expect this to be around 10%, as there are 10 different categories (numbers 0-9)

print('Random test accuracy is %1.1f%%' % accuracy)
cb_checkpoint = ModelCheckpoint(filepath='best-cnn-model.hdf5', verbose=1, save_best_only=True)



history = model.fit(X_train, y_train_matrix, batch_size=50, epochs=6, 

                    validation_split=0.2, shuffle=True, callbacks=[cb_checkpoint], verbose=1)
model.load_weights('best-cnn-model.hdf5')



# let's test our model on the test set

score = model.evaluate(X_test, y_test_matrix, verbose=0)

accuracy = score[1] * 100

print("Best Model's test accuracy is %1.1f%%" % accuracy)
# let's use that one to make our final predictions



data_test = pd.read_csv('../input/test.csv')
test = np.array(list(map(lambda x: np.reshape(x, (28,28,1)), data_test.as_matrix())))

test.shape
pred = model.predict(test, batch_size=32, verbose=1)

pred[0]
# decode the one-hot encoded predictions

predicted_labels = [ np.argmax(r, axis=0) for r in pred ]

predicted_labels
import csv



with open('submission.csv', 'w') as csvfile:

    fo = csv.writer(csvfile, delimiter=',', lineterminator='\n')

    fo.writerow(['ImageId', 'Label'])

    for (index, label) in enumerate(predicted_labels):

        fo.writerow([index + 1, label])