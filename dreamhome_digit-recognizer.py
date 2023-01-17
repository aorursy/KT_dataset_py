!pwd
import os
import pandas as pd

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_dir = '../input/'

train_path = os.path.join(data_dir, 'train.csv')
test_path = os.path.join(data_dir, 'test.csv')

# 1. load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(len(train), len(test))

y_train = train['label']
x_train = train.drop(labels=['label'], axis=1)

del train

# 2. check for null and missing values
# print(x_train.isnull().any())
# print(test.isnull().any())

# 3. normalization
x_train /= 255.
test /= 255.

# 4. reshape
x_train = x_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# 5. label encoding
y_train = to_categorical(y_train)

# 6. split training and validation set
random_seed = 2
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed)

x_train, y_train

from keras import models
from keras import layers
from keras import optimizers
from keras import losses

model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                            activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                            activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                            activation='relu'))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                            activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation="softmax"))

optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)

model.compile(
    optimizer=optimizer,
    loss=losses.categorical_crossentropy,
    metrics=['acc'])

model.summary()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)
datagen.fit(x_train)

# annealing method
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=1e-5)

# fit
epochs = 10
batch_size = 86
history = model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose=2,
    steps_per_epoch=x_train.shape[0] // batch_size,
    callbacks=[learning_rate_reduction])

# plot
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title("Training and validation accuracy")
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title("Training and validation loss")
plt.legend()
# predict results
results = model.predict(test)

# select the index with the maximum probability
results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")

submission = pd.concat(
    [pd.Series(range(1, len(test) + 1), name="ImageId"), results], axis=1)

submission.to_csv("results.csv", index=False)
