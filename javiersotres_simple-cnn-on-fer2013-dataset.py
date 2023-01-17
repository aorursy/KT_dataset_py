import pandas as pd

import cv2

import os

import numpy as np

import matplotlib.pyplot as plt

import pickle



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score



from mlxtend.plotting import plot_confusion_matrix
CATEGORIES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

X = []

y = []



def load_fer_data():

    fer_data = pd.read_csv('/kaggle/input/facialexpressionrecognition/fer2013.csv')

    for index, row in fer_data.iterrows():

        try:

            pixels=np.asarray(list(row['pixels'].split(' ')), dtype=np.uint8)

            img = pixels.reshape((48,48))

            X.append(img)

            y.append(row['emotion'])

        except Exception as e:

            pass

        

load_fer_data()
fig=plt.figure(figsize=(10,6))



for counter, img in enumerate(X[:12]):

    ax = fig.add_subplot(3,4,counter+1)

    ax.imshow(X[counter], cmap='gray')

    plt.title(CATEGORIES[y[counter]])

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)



plt.show()
X = np.array(X, dtype='float32').reshape(-1, 48, 48, 1)

X=X/255.

y = np.asarray(y)
(X_train, X_val, y_train, y_val) = train_test_split(X, y,

                                                    test_size=0.3,

                                                    random_state=42,

                                                    shuffle=True,

                                                    stratify=y)
aug_train = ImageDataGenerator(

    rotation_range=20,

    zoom_range=0.15,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.15,

    horizontal_flip=True,

    fill_mode="nearest")



generator_val = ImageDataGenerator()
aug_train.fit(X_train)



generator_val.fit(X_val)
layer_size = 64



model = Sequential()



# Input layer

model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:], padding='same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))



# Hidden layers

model.add(Conv2D(layer_size, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))



model.add(Conv2D(layer_size, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))



model.add(Conv2D(layer_size, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))



model.add(Conv2D(layer_size, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))



model.add(Conv2D(layer_size, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))



model.add(Conv2D(layer_size, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))



model.add(Flatten())

model.add(Dense(layer_size))

model.add(Activation('relu'))

model.add(Dropout(0.2))



# Output layer

model.add(Dense(7, activation='softmax'))



# Compile the model

model.compile(loss='sparse_categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

earlystop = EarlyStopping(patience=7)



history = model.fit_generator(aug_train.flow(X_train, y_train, batch_size=32),

                    validation_data=generator_val.flow(X_val, y_val, batch_size=32),

                    steps_per_epoch=len(y_train) // 32,

                    epochs=100,

                    callbacks=[earlystop, learning_rate_reduction])
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
predicted_label = model.predict(X_val)

predicted_label = np.argmax(predicted_label, axis = 1)



cm  = confusion_matrix(y_val, predicted_label)

plot_confusion_matrix(cm,figsize=(8,8), cmap=plt.cm.Blues, colorbar=True, class_names=CATEGORIES)

plt.show()