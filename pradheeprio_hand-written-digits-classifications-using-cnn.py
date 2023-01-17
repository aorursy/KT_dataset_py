import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D,ZeroPadding2D,Dropout

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)
img_width=64

img_height=64
datagen=ImageDataGenerator(1/255.0,validation_split=0.2)
train_data_generator=datagen.flow_from_directory(directory="../input/handwritten-digit-classification/database",

                                                 target_size=(img_width,img_height),

                                                 class_mode='categorical',

                                                 batch_size=16,

                                                 subset='training'

)

print(len(train_data_generator))
validation_data_generator=datagen.flow_from_directory(directory="../input/handwritten-digit-classification/database",

                                                 target_size=(img_width,img_height),

                                                 class_mode='categorical',

                                                 batch_size=16,

                                                 subset='validation'

)
train_data_generator.labels
model=Sequential()

model.add(Conv2D(30, (3, 3),input_shape=(img_width,img_height,3),activation="relu"))



model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(30, (3, 3),activation="relu"))



model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors



model.add(Dense(64))

model.add(Dropout(0.2))



model.add(Dense(10,activation="softmax"))



model.summary()

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history=model.fit_generator(generator=train_data_generator,

                            steps_per_epoch=len(train_data_generator),

                            epochs=5,

                            validation_data=validation_data_generator,

                            validation_steps=len(validation_data_generator)

                           )

model.save("weight1.h5")
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epochs')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend(['train', 'test'], loc='upper left')

plt.show()