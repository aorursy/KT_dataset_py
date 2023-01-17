from tensorflow.keras.layers import Flatten, Dense, Input, Lambda, Conv2D, MaxPooling2D
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img,ImageDataGenerator, img_to_array
import tensorflow as tf

import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt 
IMAGE_SIZE = [150,150]
train_path = '../input/intel-image-classification/seg_train/seg_train'
valid_path = '../input/intel-image-classification/seg_test/seg_test'
resent_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=IMAGE_SIZE + [3])
for layer in resent_model.layers:
    layer.trainable = False
resent_model.summary()
x = Conv2D(128, (3, 3), activation='relu')(resent_model.output)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(100,activation='relu')(x)
x = Dense(6,activation='softmax')(x)

model = Model(inputs=resent_model.input, outputs=x)
model.summary()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics = ['accuracy']
)
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    horizontal_flip = True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (150,150),
                                                 batch_size = 128,
                                                 class_mode = 'categorical')
testing_set = test_datagen.flow_from_directory(valid_path,
                                               target_size = (150,150),
                                               batch_size = 128,
                                               class_mode = 'categorical')
hist = model.fit(training_set,
                validation_data = testing_set,
                epochs = 20,
                steps_per_epoch=len(training_set),
                validation_steps=len(testing_set))

# plot the loss
plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(hist.history['accuracy'], label='train acc')
plt.plot(hist.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
from tensorflow.keras.models import load_model

model.save('model.h5')
