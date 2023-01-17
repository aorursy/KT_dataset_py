import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPool2D
train_seg = '../input/seg_train/seg_train/'

test_seg = '../input/seg_test/seg_test/'

pred_seg = '../input/seg_pred/seg_pred/'
generate = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)
training_set = generate.flow_from_directory(train_seg,

                                            target_size = (100, 100),

                                            batch_size = 14034,

                                            classes = ["buildings","forest","glacier","mountain","sea","street"],

                                            class_mode = 'categorical')

test_set = generate.flow_from_directory(test_seg,

                                        target_size = (100, 100),

                                        batch_size = 3000,

                                        classes = ["buildings","forest","glacier","mountain","sea","street"],

                                        class_mode = 'categorical')
X_train,y_train = training_set.next()

X_test,y_test = test_set.next()
model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(100,100,3), activation='relu'))

model.add(MaxPool2D(pool_size=(2,2),strides=None))

model.add(Conv2D(64,(3,3), activation='relu' , padding= "same"))

model.add(MaxPool2D(pool_size=(2,2),strides=None))

model.add(Conv2D(64,(3,3), activation='relu' , padding= "same"))

model.add(MaxPool2D(pool_size=(2,2),strides=None))

model.add(Conv2D(128,(3,3), activation='relu' , padding= "same"))

model.add(MaxPool2D(pool_size=(2,2),strides=None))

model.add(Conv2D(128,(3,3), activation='relu' , padding= "same"))

model.add(MaxPool2D(pool_size=(2,2),strides=None))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.50))



model.add(Dense(6,activation ='softmax'))
model.compile(optimizer="adam", loss= "categorical_crossentropy", metrics=["accuracy"])
model.summary()
Model = model.fit(X_train, y_train, epochs = 20, verbose=1, batch_size=500, validation_split = 0.1)
score = model.evaluate(X_test, y_test, verbose=1)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
plt.plot(Model.history['accuracy'])

plt.plot(Model.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()
plt.plot(Model.history['val_loss'])

plt.plot(Model.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Test set'], loc='upper left')

plt.show()
model.save("model.h5")

from keras.models import load_model

model = load_model('model.h5')

model.summary()