import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import cv2



upic='../input/cell-images-for-detecting-malaria/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_131.png'

apic='../input/cell-images-for-detecting-malaria/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_164.png'

plt.figure(1, figsize = (15 , 7))

plt.subplot(1 , 2 , 1)

plt.imshow(cv2.imread(upic))

plt.title('Uninfected Cell')

plt.xticks([]) , plt.yticks([])



plt.subplot(1 , 2 , 2)

plt.imshow(cv2.imread(apic))

plt.title('Infected Cell')

plt.xticks([]) , plt.yticks([])



plt.show()
width = 128

height = 128
datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
trainDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',

                                           target_size=(width,height),

                                           class_mode = 'binary',

                                           batch_size = 16,

                                           subset='training')
valDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',

                                           target_size=(width,height),

                                           class_mode = 'binary',

                                           batch_size = 16,

                                           subset='validation')
model = Sequential()

model.add(Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)))

model.add(MaxPool2D(2,2))

model.add(Dropout(0.2))



model.add(Conv2D(32,(3,3),activation='relu'))

model.add(MaxPool2D(2,2))

model.add(Dropout(0.3))



model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPool2D(2,2))

model.add(Dropout(0.3))



model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss',patience=2)
history = model.fit_generator(generator = trainDatagen,

                             steps_per_epoch = len(trainDatagen),

                              epochs =20,

                              validation_data = valDatagen,

                              validation_steps=len(valDatagen),

                             callbacks=[early_stop])
def plotLearningCurve(history,epochs):

  epochRange = range(1,epochs+1)

  plt.plot(epochRange,history.history['accuracy'])

  plt.plot(epochRange,history.history['val_accuracy'])

  plt.title('Model Accuracy')

  plt.xlabel('Epoch')

  plt.ylabel('Accuracy')

  plt.legend(['Train','Validation'],loc='upper left')

  plt.show()



  plt.plot(epochRange,history.history['loss'])

  plt.plot(epochRange,history.history['val_loss'])

  plt.title('Model Loss')

  plt.xlabel('Epoch')

  plt.ylabel('Loss')

  plt.legend(['Train','Validation'],loc='upper left')

  plt.show()
plotLearningCurve(history,7)