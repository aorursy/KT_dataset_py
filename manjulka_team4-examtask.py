# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.utils import to_categorical



from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashion-mnist_test.csv')



img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



X = np.array(data_train.iloc[:, 1:])

y = to_categorical(np.array(data_train.iloc[:, 0]))



#Here we split validation data to optimiza classifier during training

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)



#Test data



X_test = np.array(data_test.iloc[:, 1:])

y_test = to_categorical(np.array(data_test.iloc[:, 0]))



#val?



X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype('float32')

X_test = X_test.reshape(X_test.shape[0],  img_rows, img_cols , 1).astype('float32')

X_val = X_val.reshape(X_val.shape[0],  img_rows, img_cols , 1).astype('float32')



#?? 

X_train /= 255

X_test /= 255

X_val /= 255
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization



num_classes = 10

#epochs = 50



#input image dimensions

img_rows, img_cols = 28, 28



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 kernel_initializer='he_normal',

                 input_shape=input_shape))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])
history = model.fit(X_train, y_train,

          batch_size=256,

          epochs=12,

          verbose=1,

          validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)
# The predict_classes function outputs the highest probability class

# according to the trained classifier for each input example.



y_test_first = np.array(data_test.iloc[:, 0])

predicted_classes = model.predict_classes(X_test)



# Check which items we got right / wrong

correct_indices = np.nonzero(predicted_classes == y_test_first)[0]

incorrect_indices = np.nonzero(predicted_classes != y_test_first)[0]
plt.rcParams['figure.figsize'] = (10,10)



plt.figure()

for i, correct in enumerate(correct_indices[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test_first[correct]))

    

plt.figure()

for i, incorrect in enumerate(incorrect_indices[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test_first[incorrect]))