from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

import keras

import os



num_classes = 120

img_rows, img_cols = 32, 32

batch_size = 16



train_data_dir = "../input/fruits/fruits-360_dataset/fruits-360/Training"

validation_data_dir = "../input/fruits/fruits-360_dataset/fruits-360/Test"



# Apply Data Augmentation

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=30,

    width_shift_range=0.3,

    height_shift_range=0.3,

    horizontal_flip=True,

    fill_mode="nearest"

)



validation_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_rows, img_cols),

    batch_size=batch_size,

    class_mode="categorical",

    shuffle=True

)



validation_generator = validation_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_rows, img_cols),

    batch_size=batch_size,

    class_mode="categorical",

    shuffle=False

)
model = Sequential()



# Padding keeps the output the same length as the input

model.add(Conv2D(32, (3, 3), padding="same", input_shape=(img_rows, img_cols, 3)))

model.add(Activation("relu"))

model.add(Conv2D(32, (3, 3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(Conv2D(64, (3, 3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation("softmax"))



print(model.summary())
from keras.optimizers import RMSprop, SGD

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



checkpoint = ModelCheckpoint("./fruits_fresh_cnn_1.h5")



earlystop = EarlyStopping(monitor="val_loss",

                          min_delta=0,

                          patience=3,

                          verbose=1,

                          restore_best_weights=True)



reduce_lr = ReduceLROnPlateau(monitor="val_loss",

                              factor=0.2,

                              patience=3,

                              verbose=1,

                              min_delta=0.0001)



callbacks = [earlystop, checkpoint]

model.compile(loss="categorical_crossentropy",

              optimizer=RMSprop(lr=0.001),

              metrics=["accuracy"])



nb_train_samples = 60498

nb_validation_samples = 20622

epochs = 5



history = model.fit_generator(train_generator,

                              steps_per_epoch=nb_train_samples//batch_size,

                              epochs=epochs,

                              callbacks=callbacks,

                              validation_data=validation_generator,

                              validation_steps=nb_validation_samples//batch_size)
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np



y_pred = model.predict_generator(validation_generator, nb_validation_samples//batch_size+1)

y_pred = np.argmax(y_pred, axis=1)



print("Confusion Matrix:")

print(confusion_matrix(validation_generator.classes, y_pred))
from keras.models import load_model

import matplotlib.pyplot as plt

import sklearn



img_rows, img_height, img_depth = 32, 32, 3

model = load_model("./fruits_fresh_cnn_1.h5")



class_labels = validation_generator.class_indices

class_labels = {v: k for k, v in class_labels.items()}

classes = list(class_labels.values())

target_names = list(class_labels.values())



plt.figure(figsize=(20, 20))

cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)

plt.imshow(cnf_matrix, interpolation="nearest")

plt.colorbar()

tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes, rotation=90)

plt.yticks(tick_marks, classes)