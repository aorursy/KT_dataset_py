######### imports ############
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2

import tensorflow as tf
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import Sequential, Model 
from keras.utils import np_utils
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19

model = VGG19(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))
print(model.summary())
####Adding output Layer
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(12, activation="softmax")(x) 

model_final = Model(input = model.input, output = predictions)
#compling our model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

model_final.summary() #Model summary
#############    Data Augmentation
gen = ImageDataGenerator(
            rotation_range=360.,
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True)
data_dir = "../input/plant-seedlings-classification"
train_data_dir = "../input/plant-seedlings-classification/train"
train_generator = gen.flow_from_directory(
                        train_data_dir,
                        target_size = (224, 224),
                        batch_size = 16, 
                        class_mode = "categorical")
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='auto')
model_final.fit_generator(
                    train_generator,
                    epochs = 30,
                    shuffle= True,
                    callbacks = [checkpoint, early])
classes = train_generator.class_indices  
print(classes)

#Invert Mapping
classes = {v: k for k, v in classes.items()}
print(classes)



test_dir = os.path.join(data_dir, 'test')
test = []

for file in os.listdir(test_dir):
    test.append(['test/{}'.format(file), file])
 
test = pd.DataFrame(test, columns=['filepath', 'file'])
test.head(2)
test.shape

prediction = []
for filepath in test['filepath']:
    img = cv2.imread(os.path.join(data_dir,filepath))
    img = cv2.resize(img,(224,224))
    img = np.asarray(img)
    img = img.reshape(1,224,224,3)
    pred = model_final.predict(img)
    prediction.append(classes.get(pred.argmax(axis=-1)[0]))
    #Invert Mapping helps to map Label
data_dir = '../input/plant-seedlings-classification'
sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
test = test.drop(columns =['filepath']) #Remove file path from test DF

test.to_csv('submission.csv', index=False)
test.head()
