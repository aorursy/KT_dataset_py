import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import imagedatautils as ci

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping
print("Importing csv data...")

# This can take about 10 seconds

df = pd.read_csv('/kaggle/input/mnist-fashion/fashion-mnist_train.csv')

y_train=df.pop('label')

X_train=df

df = pd.read_csv('/kaggle/input/mnist-fashion/fashion-mnist_test.csv')

y_test=df.pop('label')

X_test=df

print("Finished")
X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1).astype('float32')

X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1).astype('float32')



# This divides all values in the array by 255, so each colour becomes 0-1.

X_train = X_train / 255

X_test = X_test / 255



y_train = to_categorical(y_train, 10)

y_test = to_categorical(y_test, 10)



num_pixels = X_train.shape[1] * X_train.shape[2]

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))

model.add(Dense(units=10, kernel_initializer='uniform', activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



early_stopping_monitor = EarlyStopping(patience=3)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



early_stopping_monitor = EarlyStopping(patience=3)



# epochs is the number of times it loops through the training data

epochs = 1



model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=True,

          callbacks=[early_stopping_monitor])

scores = model.evaluate(X_test, y_test, verbose=0)

print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
model.save('/kaggle/working/fashion_model')
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix



# First run some data through the model - here I am using the test data

y_pred = model.predict(X_test)

# Get a list of the predicted results

yp = [np.argmax(i) for i in y_pred]

# Get the actual results

yt = [np.argmax(i) for i in y_test]

# print a confusion matrix

cm = confusion_matrix(yt, yp)

print(cm)

import matplotlib.pyplot as plt



true_classes = range(10)

# These are the classes that we use

fashion_items = {

    0: 'T-shirt/top',

    1: 'Trouser',

    2: 'Pullover',

    3: 'Dress',

    4: 'Coat',

    5: 'Sandal',

    6: 'Shirt',

    7: 'Sneaker',

    8: 'Bag',

    9: 'Ankle boot'

}

# map one on to the other

class_labels = [fashion_items[x] for x in true_classes]



# Create the confusion matrix.

fig = plt.figure()

ax = fig.add_subplot()

cax = ax.matshow(cm)

fig.colorbar(cax)

ax.set_xticks(np.arange(len(class_labels)))

ax.set_yticks(np.arange(len(class_labels)))

ax.set_xticklabels(class_labels)

ax.set_yticklabels(class_labels)

# Rotate the tick labels and set their alignment.

plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()

import mnist_viewer

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf







# These are the categories

fashion_items = {

    0:'T-shirt/top',

    1: 'Trouser',

    2: 'Pullover',

    3: 'Dress',

    4: 'Coat',

    5: 'Sandal',

    6: 'Shirt',

    7: 'Sneaker',

    8: 'Bag',

    9: 'Ankle boot'

}



model = tf.keras.models.load_model('/kaggle/working/fashion_model')

model.summary()



# Uses 10 random images from the training data to test the model prediction.

X_predict = ci.convert_csv_source('/kaggle/input/mnist-fashion/first_fashion.csv')

X_predict = X_predict.reshape(X_predict.shape[0], 28,28,1).astype('float32')

category = model.predict(X_predict)

count=0

for i in category:

    plt.subplot(2, 5, count + 1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_predict[count].reshape(28,28))

    plt.title(fashion_items[np.argmax(i)])

    count+=1

    print("Look at my favourite: "+fashion_items[np.argmax(i)])

plt.show()



X_predict = np.rot90(X_predict, axes=(2,1))

category = model.predict(X_predict)

count=0

for i in category:

    plt.subplot(2, 5, count + 1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_predict[count].reshape(28,28))

    plt.title(fashion_items[np.argmax(i)])

    count+=1

    print("Look at my favourite: "+fashion_items[np.argmax(i)])

plt.show()
# Generate a random image and try and classify it!!

X_predict = np.asarray([np.random.random((28,28))])

X_predict = X_predict.reshape(X_predict.shape[0], 28,28,1).astype('float32')

image = X_predict.squeeze()

category=model.predict(X_predict)

plt.xticks([])

plt.yticks([])

plt.grid(False)

plt.imshow(image)

plt.title(fashion_items[np.argmax(category[0])])

plt.show()

print("Is the really my "+fashion_items[np.argmax(category[0])]+ "!")
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import imagedatautils as ci





X_football = np.asarray(ci.convert('/kaggle/input/mnistfashion/footballboot.png'))

X_football = X_football.reshape(X_football.shape[0], 28,28,1).astype('float32')

image = X_football.squeeze()

category=model.predict(X_football)

plt.xticks([])

plt.yticks([])

plt.grid(False)

plt.imshow(image)

plt.title(fashion_items[np.argmax(category[0])])

plt.show()

print("Is the really my "+fashion_items[np.argmax(category[0])]+ "!")