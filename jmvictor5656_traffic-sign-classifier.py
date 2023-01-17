import pickle

from pathlib import Path

import numpy as np

import math

#import pandas as pd

# TODO: Fill this in based on where you saved the training and testing data



PATH = Path('.')



training_file = '../input/german-traffic-sign/train.p'

validation_file = '../input/german-traffic-sign/valid.p' 

testing_file = '../input/german-traffic-sign/test.p'



with open(training_file, mode='rb') as f:

    train = pickle.load(f)

with open(validation_file, mode='rb') as f:

    valid = pickle.load(f)

with open(testing_file, mode='rb') as f:

    test = pickle.load(f)

    

X_train, y_train = train['features'], train['labels']

X_valid, y_valid = valid['features'], valid['labels']

X_test, y_test = test['features'], test['labels']
n_train = len(X_train)



# TODO: Number of validation examples

n_validation = len(X_valid)



# TODO: Number of testing examples.

n_test = len(X_test)



# TODO: What's the shape of an traffic sign image?

image_shape = X_train[0].shape



# TODO: How many unique classes/labels there are in the dataset.

n_classes = len(set(y_train))



print("Number of training examples =", n_train)

print("Number of testing examples =", n_test)

print("Image data shape =", image_shape)

print("Number of classes =", n_classes)
import pandas as pd

signnames = pd.read_csv('../input/signnames/signnames.csv')
signnames
classID_signames = list(signnames['SignName'])
### Data exploration visualization code goes here.

### Feel free to use as many code cells as needed.

import matplotlib.pyplot as plt

import numpy as np

# Visualizations will be shown in the notebook.

%matplotlib inline
train_unique_indexs = list(np.unique(y_train, return_index=True)[1])

rows = len(train_unique_indexs)//4 + 1

f = plt.figure(figsize=(20, 16))

for i, index in enumerate(train_unique_indexs, 1):

    plt.subplot(rows, 4, i)

    plt.imshow(X_train[train_unique_indexs[i-1]])

    plt.axis('off')

    plt.title(classID_signames[i-1])

    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.4)
X_train = (X_train)/255

X_valid = (X_valid)/255

X_test = (X_test)/255
from keras.utils import np_utils

from keras.layers import (Conv2D, MaxPooling2D,

                          Input, Flatten, Dense, 

                          BatchNormalization, 

                          Activation, AveragePooling2D,

                          GlobalAveragePooling2D,LeakyReLU, Dropout, Add)

from keras.models import Model

from keras import layers

from keras.regularizers import l2

from keras.callbacks import Callback
y_train = np_utils.to_categorical(y_train)

y_valid = np_utils.to_categorical(y_valid)

y_test = np_utils.to_categorical(y_test)
input_shape = (32, 32, 3)

classes = 43

X_input = Input(input_shape)
class TerminateOnBaseline(Callback):

    """Callback that terminates training when either acc or val_acc reaches a specified baseline

    """

    def __init__(self, monitor='acc', baseline=0.9):

        super(TerminateOnBaseline, self).__init__()

        self.monitor = monitor

        self.baseline = baseline



    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        acc = logs.get(self.monitor)

        if acc is not None:

            if acc >= self.baseline:

                print('Epoch %d: Reached baseline, terminating training' % (epoch))

                self.model.stop_training = True
callbacks = [TerminateOnBaseline(monitor='val_acc', baseline=0.97)]
### Resnet block o be used in the model
def resnet(X, channel):

    X_short = X

    X = Conv2D(channel, (1, 1), strides = (1, 1), kernel_initializer='he_normal',use_bias=False, kernel_regularizer=l2(1e-4))(X)

    X = Conv2D(channel, (1, 1), strides = (1, 1), kernel_initializer='he_normal',use_bias=False, kernel_regularizer=l2(1e-4))(X)

    X = BatchNormalization()(X)

    X = Add()([X, X_short])##############

    X = LeakyReLU(alpha=0.1)(X)

    return X
def simple_conv(X, channel, f, s):

    X = Conv2D(channel, (f, f), strides = (s, s), kernel_initializer='he_normal')(X)

    X = BatchNormalization()(X)

    X = LeakyReLU(alpha=0.1)(X)

    return X



def conv(X, channel, f, s):

    X = Conv2D(channel, (f, f), strides = (s, s), kernel_initializer='he_normal')(X)

    X = BatchNormalization()(X)

    X = LeakyReLU(alpha=0.1)(X)

    X = Conv2D(channel, (1, 1), strides = (1, 1), kernel_initializer='he_normal')(X)

    X = BatchNormalization()(X)

    return X
X = simple_conv(X_input, 64, 3, 2)



X = resnet(X, 64)

X = conv(X, 128, 3, 2)

X = resnet(X, 128)

X = conv(X, 256, 1, 1) # test

X = resnet(X, 256) # test

X = conv(X, 512, 3, 2)

X = resnet(X, 512)

X = conv(X, 1024, 3, 2)

X = resnet(X, 1024)



X = simple_conv(X, 128, 1, 1)

X = simple_conv(X, 128, 1, 1)



X = GlobalAveragePooling2D()(X)

X = BatchNormalization()(X) # imp

output = Dropout(0.25)(X)

output = Dense(512, activation='relu')(output)

output = BatchNormalization()(output)

output = Dropout(0.5)(output)

out = Dense(43, activation='softmax')(output)
model = Model(inputs = X_input, outputs = out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=20, batch_size=32, callbacks=callbacks)
valid = model.predict(X_valid)

valid_score = len(y_valid[y_valid.argmax(axis=1)==valid.argmax(axis=1)])/len(y_valid)

print(f"Validation Score = {valid_score*100:0.2f}%")
y_test_predict = model.predict(X_test)

test_score = len(y_test[y_test.argmax(axis=1)==y_test_predict.argmax(axis=1)])/len(y_test)

print(f"Test Score = {test_score*100:0.2f}%")
import glob
img_internet = glob.glob('../input/internet-images/*.jpg')

img_internet = np.array([plt.imread(i) for i in img_internet])

img_internet = img_internet/255
f = plt.figure(figsize=(20, 16))

for i in range(5):

    plt.subplot(1, 5, i+1)

    plt.imshow(img_internet[i])

    plt.axis('off')

    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.4)
predict_internet = model.predict(img_internet)
predict_internet_id = predict_internet.argmax(axis=1)
f = plt.figure(figsize=(20, 16))

for i in range(5):

    plt.subplot(1, 5, i+1)

    plt.imshow(img_internet[i])

    plt.title(f'predicted = {classID_signames[predict_internet_id[i]]}')

    plt.axis('off')

    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.4)
total_images = 5

correct_prediction = 2

accuracy = 2/5*100

print(f"accuracy of images found on internet = {accuracy} %")
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 

### Feel free to use as many code cells as needed.

np.sort(predict_internet, axis=1)[:,::-1][:,:5]
np.max(predict_internet, axis=1)