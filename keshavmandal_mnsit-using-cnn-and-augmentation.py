import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import tensorflow as tf





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Loading Data into Train and Test

df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')
df_train.shape
df_test.shape
df_train.columns
df_train[:5]
df_test[:1]
#Reshapping them to fit into the model

train_X = df_train.drop('label', axis=1).values.reshape(-1,28,28,1)

train_Y = df_train['label'].values.reshape(-1,1) #To hold lable values
train_Y.shape
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(train_X, train_Y, test_size = 0.1, random_state=72)
X_train.shape
X_val.shape
#Plotting some images

for i in range(9):

    img = X_train[i].reshape(28,28)

    # define subplot to show plots in 3x3 matrix

    plt.subplot(330+1+i)

    # plot raw pixel data

    plt.imshow(img,cmap='gray')
# convert from integers to floats 

train_X = tf.cast(X_train, tf.float32)

Validate_train = tf.cast(X_val, tf.float32)

# normalization to range 0-1

train_X = train_X / 255.

Validate_train = Validate_train / 255.
#To make sure we do nnot overfit the model

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),

  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(256, activation='relu'),

  tf.keras.layers.Dense(10, activation='softmax')

])
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_X, Y_train, epochs=20, validation_data=(Validate_train, Y_val)#,steps_per_epoch=6000

              ,callbacks=[callback],verbose=2

    )
test_X = df_test.values.reshape(-1,28,28,1)
test_X.shape
test_X = tf.cast(test_X, tf.float32)

test_X = test_X / 255.
#Using Image Augmenetation

from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(train_X)
# Fit the model

batch_size = 75

epochs = 30

history = model.fit_generator(datagen.flow(train_X,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (Validate_train, Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[callback])
pred = model.predict(test_X)

pred = pd.DataFrame(pred)

pred['Label'] = pred.idxmax(axis=1)

pred.head(5)
pred['index'] = list(range(1,len(pred)+1))

pred.head()
submission = pred[['index','Label']]

submission.head()
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
submission.rename(columns={'index':'ImageId'},inplace = True)

submission.head()
submission.to_csv('Submission.csv',index=False)