# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split



# we create two instances with the same arguments

data_gen_args = dict(rotation_range=30,

                     width_shift_range=0.1,

                     height_shift_range=0.1,

                     horizontal_flip=True,

                     zoom_range=0.2,

                     shear_range=0.05,

                     validation_split=0.1)



'''

data_gen_args = dict(rotation_range=0.2,

                    width_shift_range=0.05,

                    height_shift_range=0.05,

                    shear_range=0.05,

                    zoom_range=0.05,

                    horizontal_flip=True,

                    fill_mode='nearest')



'''

image_datagen = ImageDataGenerator(**data_gen_args, rescale = 1.0/255)

mask_datagen = ImageDataGenerator(**data_gen_args)



# val_image_datagen = ImageDataGenerator(rescale = 1.0/255)

# val_mask_datagen = ImageDataGenerator()



# Provide the same seed and keyword arguments to the fit and flow methods

seed = 1



image_generator = image_datagen.flow_from_directory(

    '../input/dirasimulatorroadsegment/train_data/img/',

    class_mode=None,

    color_mode='grayscale',

    target_size=(60,60),

    seed=seed, 

    subset='training')



val_image_generator = image_datagen.flow_from_directory(

    '../input/dirasimulatorroadsegment/train_data/img/',

    class_mode=None,

    color_mode='grayscale',

    target_size=(60,60),

    seed=seed,

    subset='validation')



mask_generator = mask_datagen.flow_from_directory(

    '../input/dirasimulatorroadsegment/train_data/mask/',

    class_mode=None,

    color_mode='grayscale',

    target_size=(60,60),

    seed=seed,

    subset='training')



val_mask_generator = mask_datagen.flow_from_directory(

    '../input/dirasimulatorroadsegment/train_data/mask/',

    class_mode=None,

    color_mode='grayscale',

    target_size=(60,60),

    seed=seed,

    subset='validation')



# combine generators into one which yields image and masks

'''

X_train, X_val, Y_train, Y_val = train_test_split(image_generator, mask_generator, test_size=0.1)

train_generator = zip(X_train, Y_train)

'''



train_generator = zip(image_generator, mask_generator)

val_generator = zip(val_image_generator, val_mask_generator)
from PIL import Image, ImageDraw, ImageFont

from IPython.display import display



img_names = ['road118.jpg', 'road477.jpg', 'road2450.jpg', 'road7561.jpg', 'road5964.jpg', 'road3499.jpg']

img_lst = []



for name in img_names:

    img = Image.open('../input/dirasimulatorroadsegment/train_data/img/all/'+name)

    img_lst.append(img)

    display(img)
'''

!pip install -q git+https://github.com/tensorflow/examples.git

!pip install tensorflow_datasets

'''
'''

output_channels = 1 



import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras import models

from tensorflow.keras import layers

from tensorflow.keras.models import Model

from tensorflow_examples.models.pix2pix import pix2pix

import tensorflow_datasets as tfds



base_model = MobileNetV2(input_shape=[128, 128, 3], include_top=False, weights=None)

base_model.summary()



# Use the activations of these layers



layer_names = [

    'block_1_expand_relu',   # 64x64

    'block_3_expand_relu',   # 32x32

    'block_5_project',   # 16x16

]

copied_layers = [base_model.get_layer(name).output for name in layer_names]



# Create the feature extraction model



down_stack = Model(inputs=base_model.input, outputs=copied_layers)

down_stack.trainable = False



down_stack.summary()



up_stack = [

    pix2pix.upsample(128, 3, apply_dropout=True),  # 16x16 -> 32x32

    pix2pix.upsample(64, 3, apply_dropout=True),   # 32x32 -> 64x64

]



last = layers.Conv2DTranspose(

      output_channels, 3, strides=2,

      padding='same', activation='sigmoid')  #64x64 -> 128x128



inputs = layers.Input(shape=[128, 128, 3])

x = inputs



# Downsampling through the model

skips = down_stack(x)

x = skips[-1]

skips = reversed(skips[:-1])

print(skips)

                 

# Upsampling and establishing the skip connections

for up, skip in zip(up_stack, skips):

    x = up(x)

    concat = layers.Concatenate()

    x = concat([x, skip])



y = last(x)



unet_model = Model(inputs=inputs, outputs=y)



unet_model.summary()

'''
# tf.keras.utils.plot_model(unet_model, show_shapes=True)
# unet_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])
'''

history = unet_model.fit_generator(

        train_generator,

        steps_per_epoch=100,

        epochs=5,

        validation_data=val_generator,

        validation_steps=10)

'''
'''

down_stack.trainable = True



unfeezehistory = unet_model.fit_generator(

        train_generator,

        steps_per_epoch=100,

        epochs=5,

        validation_data=val_generator,

        validation_steps=10)

'''
'''

for img in img_lst:    

    img = img.resize((128,128))

    img = Image.fromarray(np.uint8(img))

    arr = np.asarray(img)

    norm = np.array([arr/255])

    out = unet_model.predict(norm)

    out = out[0]

    out = np.rint(out)

    out = out*255

    out_3_channels = np.concatenate((out,)*3, axis=2) 

    compare = np.concatenate((arr, out_3_channels), axis=1)

    display(Image.fromarray(np.uint8(compare)))

'''

import tensorflow as tf



from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, UpSampling2D, Concatenate, SeparableConv2D

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras import models

from tensorflow.keras import layers

from tensorflow.keras.models import Model

'''

image_input = Input((128, 128, 3))



conv1 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(image_input)

conv1 = Dropout(0.2)(conv1)

conv1 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(conv1)

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



conv2 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(pool1)

conv2 = Dropout(0.2)(conv2)

conv2 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(conv2)

pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



conv3 = SeparableConv2D(128, (3, 3), strides=1, activation='relu', padding='same')(pool2)

conv3 = Dropout(0.2)(conv3)

conv3 = SeparableConv2D(128, (3, 3), strides=1, activation='relu', padding='same')(conv3)

pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)



conv4 = SeparableConv2D(256, (3, 3), strides=1, activation='relu', padding='same')(pool3)

conv4 = Dropout(0.2)(conv4)

conv4 = SeparableConv2D(256, (3, 3), strides=1, activation='relu', padding='same')(conv4)



up1 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv3])

conv5 = SeparableConv2D(128, (3, 3), strides=1, activation='relu', padding='same')(up1)

conv5 = Dropout(0.2)(conv5)

conv5 = SeparableConv2D(128, (3, 3), strides=1, activation='relu', padding='same')(conv5)



up2 = Concatenate()([UpSampling2D(size=(2, 2))(conv5), conv2])

conv6 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(up2)

conv6 = Dropout(0.2)(conv6)

conv6 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(conv6)



up3 = Concatenate()([UpSampling2D(size=(2, 2))(conv6), conv1])

conv7 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(up3)

conv7 = Dropout(0.2)(conv7)

conv7 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(conv7)



conv8 = Conv2D(1, 1, strides=1, activation='sigmoid', padding='same')(conv7)



model = Model(inputs=image_input, outputs=conv8)



model.summary()

'''
'''

image_input = Input((84,84,3))



conv1 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(image_input)

conv1 = Dropout(0.3)(conv1)

conv1 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(conv1)

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



conv2 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(pool1)

conv2 = Dropout(0.3)(conv2)

conv2 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(conv2)

pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



conv3 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(pool2)

conv3 = Dropout(0.3)(conv3)

conv3 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(conv3)



up1 = Concatenate()([UpSampling2D(size=(2, 2))(conv3), conv2])

conv4 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(up1)

conv4 = Dropout(0.3)(conv4)

conv4 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(conv4)



up2 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv1])

conv5 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(up2)

conv5 = Dropout(0.3)(conv5)

conv5 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(conv5)



conv6 = Conv2D(1, 1, strides=1, activation='sigmoid', padding='same')(conv5)

outputs = conv6

model = Model(inputs=image_input, outputs=outputs)

model.summary()

'''
'''

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('segment.h5', monitor='binary_accuracy', save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])

'''
'''

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('segment_gray.h5', monitor='binary_accuracy', save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])

'''
'''

unfeezehistory = model.fit_generator(

        train_generator,

        steps_per_epoch=100,

        epochs=30,

        validation_data=val_generator,

        validation_steps=10,

        callbacks=[checkpoint])

'''
'''

for img in img_lst:    

    img = img.resize((128,128))

    img = Image.fromarray(np.uint8(img))

    arr = np.asarray(img)

    norm = np.array([arr/255])

    out = model.predict(norm)

    out = out[0]

    out = np.rint(out)

    out = out*255

    out_3_channels = np.concatenate((out,)*3, axis=2) 

    compare = np.concatenate((arr, out_3_channels), axis=1)

    display(Image.fromarray(np.uint8(compare)))

'''
'''

unfeezehistory = model.fit_generator(

        train_generator,

        steps_per_epoch=100,

        epochs=10,

        validation_data=val_generator,

        validation_steps=10)

'''
# model = tf.keras.models.load_model('segment.h5')


'''

for img in img_lst:    

    img = img.resize((84,84))

    img = Image.fromarray(np.uint8(img))

    arr = np.asarray(img)

    norm = np.array([arr/255])

    out = model.predict(norm)

    out = out[0]

    out = np.rint(out)

    out = out*255

    out_3_channels = np.concatenate((out,)*3, axis=2) 

    compare = np.concatenate((arr, out_3_channels), axis=1)

    display(Image.fromarray(np.uint8(compare)))

'''
image_input = Input((60,60,1))



conv1 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(image_input)

conv1 = Dropout(0.3)(conv1)

conv1 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(conv1)

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



conv2 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(pool1)

conv2 = Dropout(0.3)(conv2)

conv2 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(conv2)

pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



conv3 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(pool2)

conv3 = Dropout(0.3)(conv3)

conv3 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(conv3)



up1 = Concatenate()([UpSampling2D(size=(2, 2))(conv3), conv2])

conv4 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(up1)

conv4 = Dropout(0.3)(conv4)

conv4 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(conv4)



up2 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv1])

conv5 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(up2)

conv5 = Dropout(0.3)(conv5)

conv5 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(conv5)



conv6 = Conv2D(1, 1, strides=1, activation='sigmoid', padding='same')(conv5)

outputs = conv6

model2 = Model(inputs=image_input, outputs=outputs)

model2.summary()

model2.load_weights('../input/punnet/pun_net_backup_2.h5')



from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('segment_gray.h5', monitor='val_binary_accuracy', save_best_only=True)
model2.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])

unfeezehistory = model2.fit_generator(

        train_generator,

        steps_per_epoch=100,

        epochs=30,

        validation_data=val_generator,

        validation_steps=10,

        callbacks=[checkpoint])

for img in img_lst:    

    img = img.resize((60,60))

    img = img.convert('L')

    arr = np.asarray(img)

    print(arr.shape)

    norm = arr.reshape((60,60,1))

    norm = np.array([norm/255])

    print(norm.shape)

    out = model2.predict(norm)

    out = out[0]

    out = np.rint(out)

    out = out*255

    out = out.reshape((60,60))

    #out_3_channels = np.concatenate((out,)*3, axis=2) 

    compare = np.concatenate((arr, out), axis=1)

    display(Image.fromarray(np.uint8(compare)))