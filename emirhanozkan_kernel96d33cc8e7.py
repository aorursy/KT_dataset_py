#



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



from glob import glob

import cv2

import matplotlib.pyplot as plt

%matplotlib inline
categories = sorted([p.split('/')[-1] for p in glob('../input/flowers/flowers/*')])

for c in categories:

    print('Classe {} possui {} imagens'.format(c, len(glob('../input/flowers/flowers/' + c + '/*'))))
plt.figure(figsize=(24, 6))

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        rotation_range=30,

        # Trabalhar o brilho da imagem

        # 0 = imagem totalmente preta

        # 1 = imagem original

        # >1  imagem mais clara

        brightness_range=(0.5,1.5),

        horizontal_flip=True,

        width_shift_range=0.1,

        height_shift_range=0.1,

        validation_split=0.2

)



train_generator = train_datagen.flow_from_directory(

        '../input/flowers/flowers/',

        target_size=(224, 224),

        batch_size=128,

        class_mode='categorical',

        subset='training'

        )



validation_generator = train_datagen.flow_from_directory(

        '../input/flowers/flowers/',

        target_size=(224, 224),

        batch_size=128,

        class_mode='categorical',

        subset='validation')
train_generator.class_indices

validation_generator.class_indices
plt.imshow(next(train_generator)[0][0])
from keras.applications.mobilenet_v2 import MobileNetV2

from keras.models import Model

from keras.layers import *
# No MobileNet v2 o tamanho recomendado de entrada da imagem é 224x224

# https://github.com/xiaochus/MobileNetV2



model = MobileNetV2(input_shape=(224, 224, 3), 

                    include_top=False, 

                    weights='imagenet', 

                    classes=5)



model.summary()
for l in model.layers:

    l.trainable = True
model.summary()
x = Flatten()(model.output)

x = Dense(256, activation='relu')(x)

x = Dropout(0.5)(x)

x = Dense(5, activation='softmax', trainable=True)(x)



new_model = Model(model.input, x)



new_model.compile('Adam', loss='categorical_crossentropy', metrics=['accuracy'])

new_model.summary()
model.save("keras.model")
new_model.save_weights("test.hdf5")
new_model.load_weights("test.hdf5")


new_model.fit_generator(generator=train_generator,steps_per_epoch=20,epochs=400, verbose=1)
new_model.evaluate_generator(generator=validation_generator,

                         steps=400,

                         verbose=1)
