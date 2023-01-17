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
train = '/kaggle/input/Kannada-MNIST/train.csv'

test = '/kaggle/input/Kannada-MNIST/test.csv'



dataset = pd.read_csv(train)

dataset_test = pd.read_csv(test)

dataset_test = dataset_test.drop(['id'], axis = 1)



features = dataset.drop(['label'], axis = 1)

labels = dataset['label']

dataset_test.head()
features = features.values

features_test = dataset_test.values

print(features.shape)

print(features_test.shape)



features = features.reshape((features.shape[0], 28, 28))

features_test = features_test.reshape((features_test.shape[0], 28, 28))



print(features.shape)

print(features_test.shape)
from keras.utils import np_utils



seed = 300

test_size = 0.2



# Import train_test_split

from sklearn.model_selection import train_test_split

# Split the 'features' and 'labels' data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test_size, random_state = seed, stratify=labels)

#X_test, X_valid, y_train, y_valid = train_test_split(X_test, y_test, test_size = 0.1, random_state = seed)



X_train = np.repeat(X_train[..., np.newaxis], 1, -1)

X_test = np.repeat(X_test[..., np.newaxis], 1, -1)

features_test = np.repeat(features_test[..., np.newaxis], 1, -1)



y_train = np_utils.to_categorical(np.array(y_train), 10)

y_test = np_utils.to_categorical(np.array(y_test), 10)



print("features set has {} samples.".format(features.shape))

print("Training set has {} samples.".format(X_train.shape))

print("Testing set has {} samples.".format(X_test.shape))

#print("Valid set has {} samples.".format(X_valid.shape[0]


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense, BatchNormalization

from keras.models import Sequential



model = Sequential()



model.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.4))



model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'))

model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.4))



model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint  



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



WEIGHTS_FILE = 'weights.base.hdf5'

### TODO: specify the number of epochs that you would like to use to train the model.



epochs = 100



### Do NOT modify the code below this line.



checkpointer = ModelCheckpoint(filepath='WEIGHTS_FILE', 

                             verbose=1, save_best_only=True)



history = model.fit(X_train, 

                  y_train, 

                  validation_data=(X_test, y_test),

                  epochs=epochs, 

                  batch_size=200, 

                  callbacks=[checkpointer], 

                  verbose=1)
import matplotlib.pyplot as plt

import numpy



def plot_history(history):

    # summarize history for accuracy

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    # summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
plot_history(history)
print(features_test.shape)

# get index of predicted dog breed for each image in test set

predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in features_test]



print(np.array(predictions))
!rm weights.base.hdf5
print(np.array(predictions))

predictions_df = pd.DataFrame(np.array(predictions), columns = ['Label'])

predictions_df.reset_index(level=0, inplace=True)

predictions_df.columns = ['id', 'label']

predictions_df.index += 1 

predictions_df.to_csv('predictions.csv', index = False)

predictions_df.head()