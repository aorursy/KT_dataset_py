# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.layers import Dense, Dropout

from keras.models import Model

from keras.optimizers import Adam, SGD

from keras.preprocessing.image import ImageDataGenerator, image

from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from keras import backend as K

import numpy as np

import matplotlib.pyplot as plt

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGE = True

N_CLASSES = 10

EPOCHS = 15

PATIENCE = 5

TRAIN_PATH= '../input/10-monkey-species/training/training/'

VALID_PATH = '../input/10-monkey-species/validation/validation/'

MODEL_CHECK_WEIGHT_NAME = 'resnet_monki_v1_chk.h5'
K.set_learning_phase(0)

model = ResNet50(input_shape=(224,224,3),include_top=False, weights='imagenet', pooling='avg')

K.set_learning_phase(1)

x = model.output

x = Dense(512, activation='relu')(x)

x = Dropout(0.5)(x)

x = Dense(512, activation='relu')(x)

x = Dropout(0.5)(x)

output = Dense(N_CLASSES, activation='softmax', name='custom_output')(x)

custom_resnet = Model(inputs=model.input, outputs = output)



for layer in model.layers:

    layer.trainable = False



custom_resnet.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

custom_resnet.summary()
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

traingen = datagen.flow_from_directory(TRAIN_PATH, target_size=(224,224), batch_size=32, class_mode='categorical')

validgen = datagen.flow_from_directory(VALID_PATH, target_size=(224,224), batch_size=32, class_mode='categorical', shuffle=False)
es_callback = EarlyStopping(monitor='val_acc', patience=PATIENCE, mode='max')

mc_callback = ModelCheckpoint(filepath=MODEL_CHECK_WEIGHT_NAME, monitor='val_acc', save_best_only=True, mode='max')

train_history = custom_resnet.fit_generator(traingen, steps_per_epoch=len(traingen), epochs= EPOCHS, validation_data=traingen, validation_steps=len(validgen), verbose=2, callbacks=[es_callback, mc_callback])
#custom_resnet.load_weights(MODEL_CHECK_WEIGHT_NAME)

predict = custom_resnet.predict_generator(validgen, steps=len(validgen), verbose=1)

test_labels = validgen.classes

confusion_matrix(test_labels, predict.argmax(axis=1))
accuracy_score(test_labels,predict.argmax(axis=1))
