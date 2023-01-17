import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import keras



np.random.seed(2)

random_seed = 2



from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
#Load data

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

print(train.values.shape)

print(test.values.shape)
#Divide labels and features apart

y_train = train['label']

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

y_train = to_categorical(y_train, num_classes = 10)

X_train = train.drop('label', axis=1)

del train

print(X_train.values.shape)

print(y_train.shape)
#reshape the data

X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)

print(X_train.shape)

print(test.shape)
# Normalize the data

X_train = X_train / 255.0

test = test / 255.0
#hold out some validation data.

train_size = int(.9 * X_train.shape[0])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .1, random_state=random_seed)

print(X_train.shape)

print(X_val.shape)
#Data Augmentation

datagen = keras.preprocessing.image.ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
#Prepare Model

model = keras.models.Sequential([

    keras.layers.InputLayer(input_shape=[28,28,1]),

    

    keras.layers.Conv2D(filters=32, kernel_size=[5,5], padding='Same', activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=32, kernel_size=[5,5], padding='Same', activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.MaxPool2D(pool_size=[2,2]),

    keras.layers.Dropout(.25),

    

    keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding='Same', activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding='Same', activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2]),

    keras.layers.Dropout(.25),

    

    keras.layers.Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same',  activation ='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    

    keras.layers.Flatten(),

    keras.layers.Dense(units=256, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(.3),

    keras.layers.Dense(units=10, activation='softmax'),

])

optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
#Prepare callbacks

LR_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=4, factor=.5, min_lr=.00001)

EarlyStop_callback = keras.callbacks.EarlyStopping(patience=20)
# Fit the model

batch_size = 128

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = 50, validation_data = (X_val,y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[LR_callback, EarlyStop_callback])
model.evaluate(X_val, y_val)
y_pred = model.predict(test)

y_pred = np.argmax(y_pred, axis = 1)

results = pd.Series(y_pred,name="Label")

Id = pd.Series(range(1,28001),name = "ImageId")

submission = pd.concat([Id,results],axis = 1)

submission.to_csv("MySubmission.csv",index=False)