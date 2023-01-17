import numpy as np

import pandas as pd

import datetime as dt

img_rows, img_cols = 28, 28

num_classes = 10

train = pd.read_csv("../input/train.csv")

print(train.shape)

np.random.seed(seed=1984)

# train.head()
start = dt.datetime.now()

test= pd.read_csv("../input/test.csv")

print(test.shape)

# test.head()
x = train[train.columns[1:]].values.astype('float32')

x_test = test.values.astype('float32')

y = train["label"].values.astype('int32')



from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)



print(x_train.shape, y_train.shape)

print(x_valid.shape, y_valid.shape)

print(x_test.shape)
import keras

from keras import backend as K



if K.image_data_format() == 'channels_first':

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

    x_valid = x_valid.reshape(x_valid.shape[0], 1, img_rows, img_cols)

    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

    x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)



max_p = x_train.max()

x_train = x_train / max_p

x_valid = x_valid / max_p

x_test = x_test / max_p
# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_valid = keras.utils.to_categorical(y_valid, num_classes)

print(x_train.shape, y_train.shape)

print(x_valid.shape, y_valid.shape)

print(x_test.shape)
from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=input_shape))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(optimizer=RMSprop(lr=0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy'])
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.0, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(x_train)
batch_size = 64

epochs = 15

lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)



model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),

                    steps_per_epoch=x_train.shape[0] // batch_size,

                    callbacks=[lr_reduce],

                    validation_data=(x_valid, y_valid),

                    epochs = epochs, verbose = 2)
score = model.evaluate(x_valid, y_valid, verbose=0)

print('valid loss:', score[0])

print('valid accuracy:', score[1])
# Predict the values from the validation dataset

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred,axis = 1)

ids = range(1,len(y_pred_classes)+1)

submission = pd.DataFrame(np.column_stack((ids,y_pred_classes)),columns=("ImageId","Label"))

submission.to_csv("submission.csv", index=None)
end = dt.datetime.now()

print('Total time {} s.'.format((end - start).seconds))