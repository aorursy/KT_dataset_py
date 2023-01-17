from __future__ import print_function



import os.path



import keras

from keras import applications, metrics, layers, models, regularizers, optimizers

from keras.applications import ResNet50, Xception, InceptionResNetV2

from keras.models import *

from keras.layers import *

from keras.callbacks import *

from keras.preprocessing.image import ImageDataGenerator



# Globals

BATCH_SIZE = 32   # tweak to your GPUs capacity

IMG_HEIGHT = 299   # ResNetInceptionv2 & Xception like 299, ResNet50 & VGG like 224

IMG_WIDTH = IMG_HEIGHT

CHANNELS = 3

DIMS = (IMG_HEIGHT,IMG_WIDTH,CHANNELS) # what an ugly holdover from a framework not even supported by it's authors

BEST_MODEL = 'keras.best.h5'
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    shear_range=0.1,)



train_generator = train_datagen.flow_from_directory(

    'data/train',  # this is the target directory

    target_size=(IMG_HEIGHT,IMG_WIDTH),

    batch_size=BATCH_SIZE,

    class_mode='categorical')



val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(

    'data/val',

    target_size=(IMG_HEIGHT,IMG_WIDTH), 

    batch_size=BATCH_SIZE,

    class_mode='categorical')
classes = len(train_generator.class_indices)

assert classes > 0

# sometimes "breed" gets thrown in there because i forgot to tr the first line of the csv

assert classes is len(val_generator.class_indices)

n_of_train_samples = train_generator.samples

n_of_val_samples = val_generator.samples

callbacks = [ModelCheckpoint(filepath=BEST_MODEL, verbose=0, save_best_only=True),

             EarlyStopping(monitor='val_acc', patience=3, verbose=0)]



# base_model = Xception(input_shape=DIMS, weights='imagenet', include_top=False) #~

# base_model = ResNet50(input_shape=DIMS, weights='imagenet', include_top=False)

base_model = InceptionResNetV2(input_shape=DIMS, weights='imagenet', include_top=False)

for layer in base_model.layers:

    layer.trainable = False



x = base_model.output



# RESNET50 TOP : https://github.com/keras-team/keras/blob/master/keras/applications/resnet50.py#L237-L239

# x = Flatten()(x)

# x = Dense(classes, activation='softmax', name='predictions')(x)



# XCEPTION TOP : https://github.com/keras-team/keras/blob/master/keras/applications/xception.py#L232-L234

# Inception Resnet V2 : https://github.com/keras-team/keras/blob/master/keras/applications/inception_resnet_v2.py#L332-L335

x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)

x = Dense(classes, activation='softmax', name='predictions')(x)



model = Model(inputs=base_model.input, outputs=x)
model.compile(

    loss='categorical_crossentropy',

    optimizer=optimizers.Adam(1e-3),

    metrics=['acc'])



model_out = model.fit_generator(

    train_generator,

    steps_per_epoch=n_of_train_samples//BATCH_SIZE,

    epochs=15,

    validation_data=val_generator,

    validation_steps=n_of_val_samples//BATCH_SIZE,

    verbose=0,

    callbacks=callbacks)
model.load_weights(BEST_MODEL)

model.compile(

    optimizer=optimizers.Adam(lr=1e-4,),

    loss='categorical_crossentropy',

    metrics=['acc'])



model_out = model.fit_generator(

    train_generator,

    steps_per_epoch=n_of_train_samples//BATCH_SIZE,

    epochs=60,

    validation_data=val_generator,

    validation_steps=n_of_val_samples//BATCH_SIZE,

    verbose=0,

    callbacks=callbacks)
model.load_weights(BEST_MODEL)

# print(model.summary())

# for i, layer in enumerate(model.layers):

#     print(i, layer.name)

# See model information for last convolution layer. Xception is 126.

for layer in model.layers[:126]:

    layer.trainable = False

for layer in model.layers[126:]:

    layer.trainable = True


model.compile(

    optimizer=optimizers.Adam(lr=1e-4,),

    loss='categorical_crossentropy',

    metrics=['acc'])



model_out = model.fit_generator(

    train_generator,

    steps_per_epoch=n_of_train_samples//BATCH_SIZE,

    epochs=60,

    validation_data=val_generator,

    validation_steps=n_of_val_samples//BATCH_SIZE,

    verbose=0,

    callbacks=callbacks)
# clear gpu memory

from keras.preprocessing import image

import numpy as np



def load_test_image(fpath):

    img = image.load_img(fpath, target_size=(IMG_WIDTH, IMG_HEIGHT))

    x = image.img_to_array(img)

    return x



test_labels = np.loadtxt('data/sample_submission.csv', delimiter=',', dtype=str, skiprows=1)

test_images = []

test_names = test_labels[:,0]

for test_name in test_names:

   fname = '{}.jpg'.format(test_name)

   data = load_test_image(os.path.join('data/test/', fname))

   test_images.append(data)



test_images = np.asarray(test_images)

test_images = test_images.astype('float32')

test_images /= 255

print(test_images.shape)

predictions = model.predict(test_images, verbose=1)
import pandas as pd

class_indices = sorted([ [k,v] for k, v in train_generator.class_indices.items() ], key=lambda c : c[1])

columns = [b[0] for b in class_indices]

df = pd.DataFrame(predictions,columns=columns)

df = df.assign(id = test_names)

df.to_csv("submit.csv", index=False,float_format='%.4f')

print(df.head())
predictions[1][0]