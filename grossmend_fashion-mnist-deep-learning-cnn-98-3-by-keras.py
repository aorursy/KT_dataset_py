import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from keras import layers 

from keras import models

from keras import optimizers

from keras.utils import to_categorical



from sklearn.utils import shuffle

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split



from skimage import img_as_ubyte

from skimage.transform import resize



from scipy.ndimage.interpolation import rotate
# train data

train_data = pd.read_csv('/kaggle/input/fashion-mnist_train.csv')



# test data

test_data = pd.read_csv('/kaggle/input/fashion-mnist_test.csv')
print('size train data:', train_data.shape)

print('size test data:', test_data.shape)
# show few images

plt.figure(figsize=(13,7))

for idx, img_vec in enumerate(train_data.drop('label', axis=1).values[:75]):  

    plt.subplot(5, 15, idx+1)

    plt.imshow(img_vec.reshape((28,28)),cmap=plt.cm.binary)

    plt.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.show()
# split dataset train data and test data for ML



X_data = train_data.drop('label', axis=1)

y_data = train_data['label'].copy()



X_finish = test_data.drop('label', axis=1)

y_finish = test_data['label'].copy()



print('size train data:', X_data.shape)

print('size train labels:', y_data.shape)



print('size finish test data:', X_finish.shape)

print('size finish test labels:', y_finish.shape)
# frequency occurrence train labels

plt.subplots(figsize=(11,4))

y_data.value_counts().sort_index().plot('bar', color='grey')

plt.title("Frequency Histogram of Numbers in Training Data")

plt.xlabel("Number Value")

plt.ylabel("Frequency")

plt.grid(True, alpha=0.6)
# split data train and test and convert to Keras model

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
# convert train data to Keras model

X_train = X_train.values.reshape(X_train.shape[0], 28, 28 ,1)

X_train = X_train.astype('float32') / 255



X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)

X_test = X_test.astype('float32') / 255



X_check = X_finish.values.reshape(X_finish.shape[0], 28, 28 ,1)

X_check = X_check.astype('float32') / 255
# function build model Keras

def build_model():

    # add dropout between layers

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64, (3,3), activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64,(3,3), activation='relu'))

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

    opt = optimizers.Adam(lr=0.0015, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model
# convert train labels to categorical Keras

if len(y_train.shape) == 1:

    y_train = to_categorical(y_train, 10)

    y_test = to_categorical(y_test, 10)
# train neural network

cnn = build_model()

cnn.fit(X_train,

        y_train,

        epochs=7,

        batch_size=64)
# get score test data model

test_loss, test_acc = cnn.evaluate(X_test, y_test)

test_acc
# get score finish data model

test_loss, test_acc = cnn.evaluate(X_check, to_categorical(y_finish, 10))

test_acc