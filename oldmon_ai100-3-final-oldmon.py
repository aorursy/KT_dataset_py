import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.layers import Flatten, Dense

from keras.models import Model

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
DATASET_PATH  = '/kaggle/input/ml100-03-final/image_data/train'

TEST_PATH = '/kaggle/input/ml100-03-final/image_data/test'

OUTPUT_PATH = '/kaggle/working'

NUM_CLASSES = 5

NUM_EPOCHS = 32

SEED = 77

# saved model

WEIGHTS_FINAL = 'model-InceptionResNetV2.h5'

categories=os.listdir(DATASET_PATH)
# Image preprocess

train_datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.125, height_shift_range=0.125, zoom_range=0.125, horizontal_flip=True,

                                   validation_split=0.2, rescale=1. / 255)

train_batches = train_datagen.flow_from_directory(DATASET_PATH, subset = 'training', seed = SEED)

valid_batches = train_datagen.flow_from_directory(DATASET_PATH, subset = 'validation', seed = SEED)

for cls, idx in train_batches.class_indices.items():

    print('Class #{} = {}'.format(idx, cls))
# Build Model

net = InceptionResNetV2(include_top=False, input_shape=(256, 256, 3))

x = net.output

x = Flatten()(x)

# Add Dense layer, each probability by softmax

output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

net_final = Model(inputs=net.input, outputs=output_layer)

net_final.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# Train

net_final.fit_generator(train_batches, validation_data = valid_batches, epochs = NUM_EPOCHS)

# Store Model

net_final.save(WEIGHTS_FINAL)
from keras.preprocessing import image

out = np.array(['id', 'flower_class'])

testfiles=os.listdir(TEST_PATH)

for testfile in testfiles:

    filename = testfile.split('.')[0]

    img = image.load_img(TEST_PATH+'/'+testfile,target_size=(256, 256))

    if img is None:

        continue

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis = 0)

    x = x /255.

    pred = net_final.predict(x)[0]

    tof=np.argmax(pred)

    out = np.vstack((out,[filename, tof]))



pd.DataFrame(out).to_csv(OUTPUT_PATH+'/prediction.csv',index=False,header=False)