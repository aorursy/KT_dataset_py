# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator
df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
df.head()
Y = df['label']

X = df.drop(['label'], axis = 1)
X = X / 255.0

X = X.values.reshape(-1,28,28,1)

Y = np.array(Y)

Y = to_categorical(Y, num_classes = 10)
print(X.shape)

print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = Y, random_state = 31, test_size = 0.2)
print(type(X_train), type(X_test) , type(Y_train),type( Y_test))
g = plt.imshow(X_train[0][:,:,0])
model = Sequential()
model.add(Conv2D(32,(5,5),padding = 'Same',activation ='relu', input_shape = (28,28,1)))



model.add(Conv2D(64,(3,3), padding = 'same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(128,(3,3), padding = 'same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(64,(3,3), padding = 'same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation = 'softmax'))



# Maybe we can try augementing the data to improve the accuracy

datagen = ImageDataGenerator(featurewise_center=False, 

                             samplewise_center=False, 

                             featurewise_std_normalization=False, 

                             samplewise_std_normalization=False, 

                             zca_whitening=False, 

                             zca_epsilon=1e-06, 

                             rotation_range=10, 

                             width_shift_range=0.1, 

                             height_shift_range=0.1, 

                             brightness_range=None, 

                             shear_range=0.1, 

                             zoom_range=0.15, 

                             channel_shift_range=0.0, 

                             fill_mode='nearest', 

                             cval=0.0, 

                             horizontal_flip=False, 

                             vertical_flip=False, 

                             rescale=None, 

                             preprocessing_function=None, 

                             data_format=None, validation_split=0.0, dtype=None)
datagen.fit(X_train)
model.compile(optimizer = "Nadam", loss = "categorical_crossentropy", metrics = ["accuracy"])
max_epochs = 30

batch_size = 256
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size) ,

                              epochs = max_epochs, verbose = 1, validation_data = (X_test, Y_test),

                              steps_per_epoch=X_train.shape[0] // batch_size)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'g', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'g', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
test_df.head()
test_df = test_df / 255.0

test_arr = test_df.values.reshape(-1,28,28,1)
Y_pred_arr = model.predict(test_arr)
Y_pred_arr.shape
Y_pred_arr = np.argmax(Y_pred_arr, axis = 1)
results = pd.Series(Y_pred_arr,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)