#Import tensorflow and keras library

import tensorflow as tf

import keras_preprocessing

from tensorflow.keras.preprocessing import image

import pickle

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import TensorBoard

from keras.models import Sequential

from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,Dropout

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

import keras

from tensorflow.keras.callbacks import ModelCheckpoint



print("[INFO]: Tensorflow version{}".format(tf.__version__))

state_gpu = tf.test.gpu_device_name()

print("[INFO]: GPU usage{0}".format(state_gpu))
ROT_RANGE = 10



# this is the augmentation configuration we will use for training

train_gen = ImageDataGenerator(

rescale = 1./255,

rotation_range = ROT_RANGE,

width_shift_range=0.2,

height_shift_range=0.2,

shear_range=0.2,

zoom_range=0.2,

horizontal_flip=True,

fill_mode='nearest')



valid_gen = ImageDataGenerator(rescale = 1./255)

TRAINING_DIR = '../input/tomato/New Plant Diseases Dataset(Augmented)/train/' 

VALIDATION_DIR = '../input/tomato/New Plant Diseases Dataset(Augmented)/valid/'
TARGET_SIZE = (224,224)

TRAIN_BATCH_SIZE = 128

VALID_BATCH_SIZE = 32

SEED = 42



#Data Iterator

train_data = train_gen.flow_from_directory(

TRAINING_DIR,

target_size = TARGET_SIZE,

class_mode = 'categorical',

color_mode = "rgb",

batch_size = TRAIN_BATCH_SIZE,

shuffle = True,

seed = SEED

)



valid_data = valid_gen.flow_from_directory(

VALIDATION_DIR,

target_size = TARGET_SIZE,

class_mode = 'categorical',

color_mode = "rgb",

batch_size = VALID_BATCH_SIZE

)
from keras.applications.vgg16 import VGG16

base_model_weights_path = '/kaggle/input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model = VGG16(weights=base_model_weights_path, include_top=False, input_shape=(224, 224, 3))



x = keras.layers.Flatten() (base_model.output)

x = keras.layers.Dense(256, activation="relu")(x)

x = keras.layers.Dropout(0.25)(x)

output = keras.layers.Dense(10, activation='softmax')(x)

model = keras.models.Model(inputs=base_model.input, outputs=output)



# The newly added layers are initialized with random values.

# Make sure based model remain unchanged until newly added layers weights get reasonable values.

for layer in base_model.layers:

    layer.trainable = False
model.summary()
LEARNING_RATE = 0.0001

#LEARNING_RATE = 0.001



#Optimizer

opt = Adam(lr = LEARNING_RATE)

model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
TRAINING_NUM = train_data.n #or train_data.samples

VALID_NUM = valid_data.n

EPOCHS = 25



STEP_SIZE_TRAIN = TRAINING_NUM // TRAIN_BATCH_SIZE 

STEP_SIZE_VALID = VALID_NUM // VALID_BATCH_SIZE



# Fit model to get reasonable weights for newly added layers.

history = model.fit_generator(generator = train_data,

                             steps_per_epoch = STEP_SIZE_TRAIN,

                             validation_data = valid_data,

                             validation_steps = STEP_SIZE_VALID,

                             epochs = EPOCHS)
import matplotlib.pyplot as plt



plt.figure( figsize = (15,8)) 

    

plt.subplot(221)  

# Accuracy 

plt.plot(model.history.history['accuracy'])

plt.plot(model.history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()





# Loss

plt.figure( figsize = (15,8)) 

plt.subplot(222)  

plt.plot(model.history.history['loss'])

plt.plot(model.history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()
best_val_acc = max(history.history['val_accuracy'])

print("[INFO] Best Validation Accuracy: %",best_val_acc*100)
for layer in base_model.layers:

    layer.trainable = True



# compile the model with a SGD/momentum optimizer

# and a very slow learning rate (This ensures the base model weights do not change a lot)

model.compile(loss='categorical_crossentropy',

              optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])
TRAINING_NUM = train_data.n #or train_data.samples

VALID_NUM = valid_data.n

EPOCHS = 25



STEP_SIZE_TRAIN = TRAINING_NUM // TRAIN_BATCH_SIZE 

STEP_SIZE_VALID = VALID_NUM // VALID_BATCH_SIZE



# Fit model to get reasonable weights for newly added layers.

history = model.fit_generator(generator = train_data,

                             steps_per_epoch = STEP_SIZE_TRAIN,

                             validation_data = valid_data,

                             validation_steps = STEP_SIZE_VALID,

                             epochs = EPOCHS)
best_val_acc = max(history.history['val_accuracy'])

print("[INFO] Best Validation Accuracy: %",best_val_acc*100)
import matplotlib.pyplot as plt



plt.figure( figsize = (15,8)) 

    

plt.subplot(221)  

# Accuracy 

plt.plot(model.history.history['accuracy'])

plt.plot(model.history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()





# Loss

plt.figure( figsize = (15,8)) 

plt.subplot(222)  

plt.plot(model.history.history['loss'])

plt.plot(model.history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()