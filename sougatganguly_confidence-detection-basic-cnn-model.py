import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import os

%matplotlib inline



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D

from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.utils import plot_model



from IPython.display import SVG, Image

from livelossplot import PlotLossesKerasTF

import tensorflow as tf

print("Tensorflow version:", tf.__version__)
img = plt.imread('../input/confident-unconfident/confident-unconfident/test/confident/PrivateTest_10516065.jpg')

plt.imshow(img)
for expression in os.listdir("../input/confident-unconfident/confident-unconfident/train"):

    print(str(len(os.listdir("../input/confident-unconfident/confident-unconfident/train/" + expression))) + " " + expression + " images")
img_size = 48

batch_size = 64



datagen_train = ImageDataGenerator(horizontal_flip=True)



train_generator = datagen_train.flow_from_directory("../input/confident-unconfident/confident-unconfident/train/",

                                                    target_size=(img_size,img_size),

                                                    color_mode="grayscale",

                                                    batch_size=batch_size,

                                                    class_mode='binary',

                                                    shuffle=True)



datagen_validation = ImageDataGenerator(horizontal_flip=True)

validation_generator = datagen_validation.flow_from_directory("../input/confident-unconfident/confident-unconfident/test/",

                                                    target_size=(img_size,img_size),

                                                    color_mode="grayscale",

                                                    batch_size=batch_size,

                                                    class_mode='binary',

                                                    shuffle=False)
model = Sequential()



# 1 - Convolution layer

model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48, 1)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# 2nd Convolution layer

model.add(Conv2D(128,(5,5), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# 3rd Convolution layer

model.add(Conv2D(512,(3,3), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# 4th Convolution layer

model.add(Conv2D(512,(3,3), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# Flattening

model.add(Flatten())



# Fully connected layer 1st layer

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))



# Fully connected layer 2nd layer

model.add(Dense(512))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))



model.add(Dense(1, activation='sigmoid'))



opt = Adam(lr=0.0005)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
%%time



epochs = 15

steps_per_epoch = train_generator.n//train_generator.batch_size

validation_steps = validation_generator.n//validation_generator.batch_size



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,

                              patience=2, min_lr=0.00001, mode='auto')

checkpoint = ModelCheckpoint("model_weights_new.h5", monitor='val_accuracy',

                             save_weights_only=True, mode='max', verbose=1)

callbacks = [PlotLossesKerasTF(), checkpoint, reduce_lr]



history = model.fit(

    x=train_generator,

    steps_per_epoch=steps_per_epoch,

    epochs=epochs,

    validation_data = validation_generator,

    validation_steps = validation_steps,

    callbacks=callbacks

)
'''model_json = model.to_json()

with open("model_new.json", "w") as json_file:

    json_file.write(model_json)'''