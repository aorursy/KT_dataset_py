import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.utils import np_utils

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.datasets import mnist

import matplotlib.pyplot as plt

import matplotlib.cm as cm

%matplotlib inline
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train = X_train/255.0

test = test/255.0



X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
# plotting the first five training images

fig = plt.figure(figsize=(20,20))

for i in range(5):

    ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])

    ax.imshow(X_train[i].reshape(28,28), cmap='gray')

    ax.set_title(str(Y_train[i]))
Y_train = np_utils.to_categorical(Y_train, num_classes=10)



# print the first five encoded training labels

print('One-hot Encoded labels:')

print(Y_train[:10])
random_seed = 69

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

print(X_train.shape, Y_train.shape)
model = Sequential()



model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(BatchNormalization())



model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())



model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 256, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.3))



model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'Same',  activation ='relu'))

model.add(Conv2D(filters = 512, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 256, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.3))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Dense(10, activation = "softmax"))

model.summary()
from keras.utils import plot_model

from IPython.display import Image



plot_model(model, to_file='keras_CNN_model.png', show_shapes=True, show_layer_names=True)

Image("keras_CNN_model.png")
# Define Optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_acc', 

                              patience=3, 

                              verbose=1, 

                              factor=0.5, 

                              min_lr=0.00001)



hist = model.fit(X_train, 

                 Y_train, 

                 batch_size=128, 

                 epochs=10,

                 validation_data=(X_val, Y_val), 

                 callbacks=[reduce_lr],

                 verbose=1, 

                 shuffle=True)
Y_test = model.predict_classes(test, verbose=2)
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission['Label'] = Y_test

submission.to_csv('submission.csv',index=False)