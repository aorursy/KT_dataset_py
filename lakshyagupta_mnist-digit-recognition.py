# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import csv

import os

import cv2

import matplotlib.pyplot as plt



from keras.utils import to_categorical



csv_file = os.listdir("../input")

print(csv_file)

# Any results you write to the current directory are saved as output.
# Read train.csv



features = []

with open('../input/train.csv','rt') as f:

    data = csv.reader(f)

    for row in data:

        features.append(np.array(row[:]))

key = features[0]

features = np.array(features[1:])

numbers = features[:,0]

categories = list(set(numbers))

features = features[:,1:]



print('Labels -', categories)

print('Training Data Size -', features.shape)
# Reshape training data



images = []

for sample in features:

    temp = np.reshape(sample, (28, 28, 1))

    #ret, temp = cv2.threshold(temp.astype(np.float), 120, 255, cv2.THRESH_BINARY)

    #temp = np.reshape(temp, (28, 28, 1))

    #kernel = np.ones((1,1),np.uint8)

    #erosion = cv2.erode(thresh, kernel, iterations = 1)

    images.append(temp)

images = np.array(images)

labels = to_categorical(numbers)



print('Reshaped Training Data Size -', images.shape, labels.shape)
# Split dataset



train_split = 0.8

xtrain = images[:int(train_split*42000)]

ytrain = labels[:int(train_split*42000)]

xval = images[int(train_split*42000):]

yval = labels[int(train_split*42000):]



print('Training Data -', xtrain.shape, ytrain.shape)

print('Validation Data -', xval.shape, yval.shape)
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Dropout, Flatten

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator



model = Sequential()

model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(28,28,1)))

#model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=3, activation='relu'))

#model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=3, activation='relu'))

#model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=3, activation='relu'))

#model.add(Dropout(0.5))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

#model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))



datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, 

                             featurewise_std_normalization=False, samplewise_std_normalization=False, 

                             zca_epsilon=1e-06, rotation_range=0.2, 

                             width_shift_range=0.1, height_shift_range=0.1, brightness_range=None, 

                             shear_range=0.1, zoom_range=0.2, fill_mode='nearest', cval=0.0, rescale=None)



datagen.fit(images)



opt = optimizers.SGD(lr=0.001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history  = model.fit_generator(datagen.flow(images, labels, batch_size=32), steps_per_epoch=len(images)/32, epochs=40, 

                               validation_data = (xval, yval))
plt.plot(history.history['acc'])

plt.plot(history.history['loss'])

plt.title('Training')

plt.ylabel('Score')

plt.xlabel('Epoch')

plt.legend(['Accuracy', 'Loss'], loc='upper left')

plt.show()
plt.plot(history.history['val_acc'])

plt.plot(history.history['val_loss'])

plt.title('Validation')

plt.ylabel('Score')

plt.xlabel('Epoch')

plt.legend(['Accuracy', 'Loss'], loc='upper left')

plt.show()
# Read test.csv



features = []

with open('../input/test.csv','rt') as f:

    data = csv.reader(f)

    for row in data:

        features.append(np.array(row[:]))

key = features[0]

features = np.array(features)

features = features[1:,:]



print('Testing Data Size -', features.shape)
# Reshape testing data



images = []

for sample in features:

    temp = np.reshape(sample, (28, 28, 1))

    #ret, temp = cv2.threshold(temp.astype(np.float), 120, 255, cv2.THRESH_BINARY)

    #temp = np.reshape(temp, (28, 28, 1))

    #kernel = np.ones((1,1),np.uint8)

    #erosion = cv2.erode(thresh, kernel, iterations = 1)

    images.append(temp)

images = np.array(images)



print('Reshaped Training Data Size -', images.shape)
pred = model.predict(images, batch_size=None, verbose=1)
final_pred = []

for pred_i in pred:

    final_pred.append(numbers[np.argmax(pred_i)])





print(len(final_pred))
submit_list = [['ImageID', 'Label']]

for i in range(len(final_pred)):

        submit_list.append(([str(i+1), final_pred[i]]))

#print(submit_list)





with open('submit.csv', 'w') as writeFile:

    writer = csv.writer(writeFile)

    writer.writerows(submit_list)