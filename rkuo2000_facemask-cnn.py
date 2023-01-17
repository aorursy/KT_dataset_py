import os

import numpy as np

train_dir = "../input/face-mask-12k-images-dataset/Face Mask Dataset/Train/"

valid_dir = "../input/face-mask-12k-images-dataset/Face Mask Dataset/Validation/"

test_dir  = "../input/face-mask-12k-images-dataset/Face Mask Dataset/Test/"
from tensorflow.keras.preprocessing.image import ImageDataGenerator



target_size=(224,224)

batch_size = 16
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=True)



train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=target_size,

    batch_size=batch_size,

    color_mode='rgb',    

    shuffle=True,

    seed=42,

    class_mode='categorical')
valid_datagen = ImageDataGenerator(rescale=1./255)



valid_generator = valid_datagen.flow_from_directory(

    valid_dir,

    target_size=target_size,

    batch_size=batch_size,

    color_mode='rgb',

    shuffle=False,    

    class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=target_size,

    batch_size=batch_size,

    color_mode='rgb', 

    shuffle=False,    

    class_mode='categorical')
import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.models import Model,save_model

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import ModelCheckpoint
num_classes = 2 # WithMask, WithoutMask
input_shape = (224,224,3)
# Build Model

input_image = Input(shape=input_shape)

# 1st Conv layer

model = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape)(input_image)

model = MaxPooling2D((2, 2),padding='same')(model)

# 2nd Conv layer

model = Conv2D(32, (3, 3), activation='relu', padding='same')(model)

model = MaxPooling2D((2, 2),padding='same')(model)

# 3rd Conv layer

model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)

model = MaxPooling2D((2, 2),padding='same')(model)

# 4th Conv layer

model = Conv2D(128, (3, 3), activation='relu', padding='same')(model)

model = MaxPooling2D((2, 2),padding='same')(model)

# 5th Conv layer

model = Conv2D(256, (3, 3), activation='relu', padding='same')(model)

model = MaxPooling2D((2, 2),padding='same')(model)

# FC layers

model = Flatten()(model)

#model = Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(model)

model = Dense(1024)(model)

#model = Dropout(0.2)(model)



#model = Dense(64, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(model)

model = Dense(64)(model)

#model = Dropout(0.2)(model)



output= Dense(num_classes, activation='softmax')(model)



model = Model(inputs=[input_image], outputs=[output])



model.summary()
# Compile Model

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST =test_generator.n//test_generator.batch_size

num_epochs = 50
# Train Model

model.fit_generator(train_generator,steps_per_epoch=STEP_SIZE_TRAIN,epochs=num_epochs, validation_data=valid_generator, validation_steps=STEP_SIZE_VALID) #, callbacks=[checkpoint])
save_model(model, "facemask_cnn.h5")
score = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)

print(score)
from sklearn.metrics import classification_report, confusion_matrix
predY=model.predict_generator(test_generator)

y_pred = np.argmax(predY,axis=1)

#y_label= [labels[k] for k in y_pred]

y_actual = test_generator.classes



cm = confusion_matrix(y_actual, y_pred)

print(cm)
# report

labels = ['withMask', 'withoutMask']

print(classification_report(y_actual, y_pred, target_names=labels))