# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from os.path import join



digit_recognizer_dir = '../input/digit-recognizer'

train = pd.read_csv(join(digit_recognizer_dir, 'train.csv'))

test = pd.read_csv(join(digit_recognizer_dir, 'test.csv'))



train.head()
Y_train = train['label']

X_train = train.drop(labels=['label'], axis=1)



#normalize

X_train = X_train / 255

X_test = test / 255
import matplotlib.pyplot as plt

import matplotlib.image as mpimg



fig, axs = plt.subplots(1, 7, figsize=(20, 5))

for n in range(0,7):

    array = X_train.iloc[n].values

    axs[n].set_title('Label = ' + str(Y_train[n]))

    axs[n].imshow(array.reshape(28,28))
distribution = Y_train.value_counts().sort_index()

plt.bar(distribution.index, distribution.values)
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split



X_train = X_train.values.reshape(-1, 28,28,1)

X_test = X_test.values.reshape(-1, 28,28,1)



# Encode labels to one hot vectors

Y_train = to_categorical(Y_train, num_classes = 10)



#Split data for training and validation

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=1)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



#generate more data by rotate / zooming / shifting

data_generator = ImageDataGenerator(rotation_range=10,

                                     zoom_range = 0.2,

                                     width_shift_range = 0.1,

                                     height_shift_range = 0.1)



data_generator.fit(X_train)
from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D



model = Sequential()



model.add(Conv2D(filters=32, kernel_size = (5,5), padding='same', activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Dropout(0.25))



model.add(Conv2D(filters=64, kernel_size = (5,5), padding='same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Dropout(0.25))



model.add(Conv2D(filters=128, kernel_size = (3,3), padding='same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])

model.summary()
history = model.fit(data_generator.flow(X_train,Y_train, batch_size=128),

                    epochs = 6,

                    steps_per_epoch=X_train.shape[0] // 128,

                    validation_data = (X_val, Y_val),

                    verbose = 1)
#training history



fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], label="Training loss")

ax[0].plot(history.history['val_loss'], label="validation loss",)

legend = ax[0].legend(loc='best')



ax[1].plot(history.history['accuracy'], label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], label="Validation accuracy")

legend = ax[1].legend(loc='best')
from tensorflow import math

import seaborn as sn



#make predictions on X_val

predictions = model.predict_classes(X_val)



#reverse to_categorical

Y_truth = [np.argmax(x) for x in Y_val]



#confusion matrix

conf_mat = math.confusion_matrix(Y_truth, predictions)



sn.heatmap(conf_mat, annot=True, annot_kws={"size": 11}, fmt='d') # font size

plt.show()
from sklearn.metrics import accuracy_score



print("good predictions : " + str(accuracy_score(Y_truth, predictions, normalize=False)))

print("total : " + str(len(Y_truth)))

print("Accuracy score : " + str(accuracy_score(Y_truth, predictions)))
#figure out what are our prediction errors

fig, axs = plt.subplots(3, 10, figsize=(20, 9))

i = 0

j = 0

n = 0

for truth, pred in zip(Y_truth, predictions):

    if truth != pred:

        axs[j][i].set_title('truth: ' + str(truth) + '\npredict:' + str(pred))

        axs[j][i].imshow(X_val[n].reshape(28,28))

        i += 1

        if i == 10 and j == 2:

            break

        if i == 10:

            i = 0

            j += 1

    n += 1
example = pd.read_csv(join(digit_recognizer_dir, 'sample_submission.csv'))

print(example)

submission = model.predict(X_test)

submission = np.argmax(submission, axis=1)

submission = pd.Series(submission, name="Label")

submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), submission], axis=1)

print(submission)
submission.to_csv("submission.csv", index=False)