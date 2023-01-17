from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import Input,Dense,Dropout,GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
#                                    rotation_range=30,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True,
#                                    fill_mode='nearest')

# test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
# training_set=train_datagen.flow_from_directory(directory='dataset/training_set',target_size=(299,299),batch_size=32)
# test_set=test_datagen.flow_from_directory('dataset/test_set/',target_size=(299,299),batch_size=32)
# inception=InceptionV3(include_top=False,weights='imagenet')

# x1=inception.output
# x2=GlobalAveragePooling2D()(x1)
# x3=Dense(1024,activation='relu')(x2)

# predictions=Dense(2,activation='softmax')(x3)

# model=Model(inputs=inception.input,outputs=predictions)
# for layer in inception.layers:
#     layer.trainable=False
# layer_to_freeze=250
# for layer in model.layers[:layer_to_freeze]:
#     layer.trainable=False
    
# for layer in model.layers[layer_to_freeze:]:
#     layer.trainable=True
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# history=model.fit_generator(training_set,
#                            epochs=2,
#                            steps_per_epoch=8000//32,
#                            validation_data=test_set,
#                            validation_steps=2000//32,
#                            class_weight='auto')