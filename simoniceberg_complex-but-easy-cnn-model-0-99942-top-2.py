import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv') 
test = pd.read_csv('../input/test.csv')

Y_train = train['label']
X_train = train.drop(labels = ['label'], axis = 1)
X_train = X_train / 255.0
test = test / 255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)
Y_train = to_categorical(Y_train, num_classes = 10)
random_seed = 10
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = random_seed)
model = Sequential()

model.add(Conv2D(
    filters = 64,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu',
    input_shape = (28, 28, 1)))

model.add(Conv2D(
    filters = 128,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(
    filters = 128,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 128,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 128,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 256,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(
    filters = 256,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 256,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 256,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 512,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu'))

model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(
    filters = 512,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 512,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 512,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 512,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu'))

#This 1 x 1 conv layer is added to increase depth and add some non-linearity to the model. 1 x 1 conv layers were introduced in Google Inception Net,
#which shown great results.
model.add(Conv2D(
    filters = 1024,
    kernel_size = (1, 1),
    padding = 'same',
    activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation = 'softmax'))
optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08) 

#compile the model
#for loss, I chose categorical crossentropy
#it should be noted that categorical crossentropy can only be used for categorical data
#that's why the labels were transformed in the code above
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

epochs = 1 #you really need to change this, this is just for convinience
batch_size = 128
datagen = ImageDataGenerator(
    rotation_range = 10, #randomly rotates the image by 10 degrees
    zoom_range = 0.1, #randomly zooms the image by 10%
    width_shift_range = 0.1, #randomly shifts the image horizontally by 10% of the width
    height_shift_range = 0.1 #randomly shifts the image vertically by 10% of the height
)
history = model.fit_generator(datagen.flow(x = X_train, y = Y_train, batch_size = batch_size),
    epochs = epochs, validation_data = (X_val, Y_val),
    verbose = 2, steps_per_epoch = X_train.shape[0] // batch_size)
results = model.predict(test)

results = np.argmax(results, axis = 1)

results = pd.Series(results, name = "Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), results], axis = 1)

submission.to_csv("cnn_mnist_datagen.csv", index = False)