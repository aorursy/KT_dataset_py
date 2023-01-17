import numpy as np

import pandas as pd

import tensorflow as tf

from skimage import io, transform

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score
from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau

from keras.datasets import mnist
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

print(train_data.shape,

     test_data.shape)
(x_train1, y_train1), (x_test1, y_test1) = mnist.load_data()

x_train1 = np.concatenate((x_test1, x_train1))

y_train1 = np.concatenate((y_test1, y_train1))



x_train1 = x_train1.reshape((x_train1.shape[0], 28, 28, 1))

print(x_train1.shape, y_train1.shape)
x = np.array(train_data.drop(['label'], axis = 1))

y = np.array(train_data['label'])

test_data = np.array(test_data)



x = x.reshape((x.shape[0], 28, 28, 1))

test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)



x = np.concatenate((x, x_train1))

y = np.concatenate((y, y_train1))



x = x/255

test_data = test_data/255

y = to_categorical(y, num_classes = 10)



print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10, shuffle = True)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (3,3), activation ='relu'))

model.add(MaxPool2D((2,2)))



model.add(BatchNormalization())



model.add(Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'))



model.add(BatchNormalization())



model.add(Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'))



model.add(BatchNormalization())



model.add(Flatten())



model.add(Dense(128, activation = "relu"))

model.add(Dropout(0.30))

model.add(Dense(10, activation = "softmax"))





optimizer = RMSprop(lr = 0.01, rho = 0.9, epsilon = 1e-08, decay = 0.0)



model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])



model.summary()
datagen = ImageDataGenerator(

        rotation_range = 10,

        zoom_range = 0.1,

        width_shift_range = 0.1,

        height_shift_range = 0.1,)



train_batch = datagen.flow(x, y, batch_size = 64)

val_batch = datagen.flow(x_test, y_test, batch_size = 64)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_loss', 

                                            patience = 3, 

                                            verbose = 1, 

                                            factor = 0.1, 

                                            min_lr = 0.00001)
history = model.fit_generator(generator = train_batch,

                              epochs = 100, 

                              steps_per_epoch = len(train_batch),

                              validation_data = val_batch,

                              validation_steps = len(val_batch),

                              verbose = 1,

                             callbacks = [learning_rate_reduction])
res = model.predict_classes(test_data, batch_size = 64)

result = pd.Series(res, name = 'Label')

submission = pd.concat([pd.Series(range(1, 28001), name = 'ImageId'), result], axis = 1)

submission.to_csv('Submission.csv', index = False)
res = model.evaluate(x, y, batch_size = 1024)

print(res[1]*100)