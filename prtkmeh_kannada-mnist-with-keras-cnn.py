# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.optimizers import Adam

from keras.utils import np_utils

from matplotlib import pyplot as plt

from sklearn.metrics import classification_report,confusion_matrix
#Read and seperate data and labels/IDs

train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

train_Y = train_data['label']

train_data = train_data.drop(columns=['label'])

test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

test_id = test_data['id']

test_data = test_data.drop(columns=['id'])

train_X = train_data.values

test_X = test_data.values

print(train_X.shape, train_Y.shape, test_X.shape)
qw = 5

plt.imshow(train_X[qw,:].reshape(28,28),cmap='gray')

plt.title(train_Y[qw])

plt.xticks([])

plt.yticks([])

plt.show()
#Number of each class

train_Y.value_counts(sort=False)
#Restructure data

num = 28

train_X_re = train_X.reshape((-1, num, num, 1)).astype('float32')  

test_X_re = test_X.reshape((-1, num, num, 1)).astype('float32')

print(train_X_re.shape, test_X_re.shape)
#One hot encode

number_of_classes = 10

train_Y = np_utils.to_categorical(train_Y, number_of_classes)
# normalize inputs from 0-255 to 0-1

train_X_re/=255

test_X_re/=255
# create model

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(train_X_re.shape[1], train_X_re.shape[2], 1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(number_of_classes, activation='softmax'))



model.summary()
# Compile model

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# Train Test split

from sklearn.model_selection import train_test_split

train_X_t,train_X_ts,train_Y_t,train_Y_ts = train_test_split(train_X_re, train_Y,test_size=0.2)
print(train_X_t.shape)

print(train_Y_t.shape)

print(train_X_ts.shape)

print(train_Y_ts.shape)
# Fit the model

model_result=model.fit(train_X_t, train_Y_t, validation_data=(train_X_ts, train_Y_ts), epochs=15, batch_size=200)
# plot and visualise the training and validation losses

plt.figure(figsize=(20, 10))



plt.subplot(1, 2, 1)

plt.plot(model_result.history["loss"], label="training")

plt.plot(model_result.history["val_loss"], label="validation")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(model_result.history["acc"], label="training")

plt.plot(model_result.history["val_acc"], label="validation")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()



plt.show()

#Create Classification_report and Confusion Matrix

Y_pred = model.predict_classes(train_X_re)

target_names = ['0','1','2','3','4','5','6','7','8','9']

print(classification_report(np.argmax(train_Y,axis=1), Y_pred, target_names=target_names))

cm = confusion_matrix(np.argmax(train_Y,axis=1), Y_pred)





fig, ax = plt.subplots(figsize=(8,8))

im = ax.imshow(cm)



# We want to show all ticks...

ax.set_xticks(np.arange(len(target_names)))

ax.set_yticks(np.arange(len(target_names)))

# ... and label them with the respective list entries

ax.set_xticklabels(target_names)

ax.set_yticklabels(target_names)



# Rotate the tick labels and set their alignment.

plt.setp(ax.get_xticklabels(), ha="center",va="center", color="w")



# Loop over data dimensions and create text annotations.

for i in range(len(target_names)):

    for j in range(len(target_names)):

        text = ax.text(j, i, cm[i, j],

                       ha="center", va="center", color="w")



ax.set_title("Confusion Matrix")

plt.show()
# predict on test set

predictions = model.predict_classes(test_X_re)
# create submission file

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
submission['label'] = predictions
# generate submission file in csv format

submission.to_csv('submission.csv', index=False)