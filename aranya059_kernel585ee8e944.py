import numpy as np

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

DATASET_PATH = "/kaggle/input/Corona detection/train"

test_dir = "/kaggle/input/Corona detection/test"

IMAGE_SIZE = (300, 300)

NUM_CLASSES = 3

BATCH_SIZE = 10 # try reducing batch size or freeze more layers if your GPU runs out of memory

NUM_EPOCHS = 100

LEARNING_RATE =0.0005 
train_datagen = ImageDataGenerator(rescale=1./255,

 rotation_range=50,

 featurewise_center = True,

 featurewise_std_normalization = True,

 width_shift_range=0.2,

 height_shift_range=0.2,

 shear_range=0.25,

 zoom_range=0.1,

 zca_whitening = True,

 channel_shift_range = 20,

 horizontal_flip = True ,

 vertical_flip = True ,

 validation_split = 0.2,

 fill_mode="constant")

train_batches = train_datagen.flow_from_directory(DATASET_PATH,

 target_size=IMAGE_SIZE,

 shuffle=True,

 batch_size=BATCH_SIZE,

 subset = 'training',

 seed=42,

 class_mode='categorical',

 

 )

valid_batches = train_datagen.flow_from_directory(DATASET_PATH,

 target_size=IMAGE_SIZE,

 shuffle=True,

 batch_size=BATCH_SIZE,

 subset = 'validation',

 seed=42,

 class_mode='categorical',

 

 

 )
from keras import models

from keras import layers

from keras.applications import VGG16

from keras import optimizers

from keras.layers.core import Flatten, Dense, Dropout, Lambda

conv_base = VGG16(weights="imagenet",

 include_top=False,

 input_shape=(300, 300, 3))

conv_base.trainable = False

model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(128, activation="relu"))

model.add(layers.Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy",

 

 optimizer=optimizers.Adam(lr=LEARNING_RATE),

 metrics=["acc"])
STEP_SIZE_TRAIN=train_batches.n//train_batches.batch_size

STEP_SIZE_VALID=valid_batches.n//valid_batches.batch_size

result=model.fit_generator(train_batches,

 steps_per_epoch =STEP_SIZE_TRAIN,

 validation_data = valid_batches,

 validation_steps = STEP_SIZE_VALID,

 epochs= NUM_EPOCHS,

 )
import matplotlib.pyplot as plt

def plot_acc_loss(result, epochs):

 acc = result.history["acc"]

 loss = result.history["loss"]

 val_acc = result.history["val_acc"]

 val_loss = result.history["val_loss"]

 plt.figure(figsize=(15, 5))

 plt.subplot(121)

 plt.plot(range(1,epochs), acc[1:], label="Train_acc")

 plt.plot(range(1,epochs), val_acc[1:], label="Test_acc")

 plt.title("Accuracy over " + str(epochs) + "  Epochs", size=15)

 plt.legend()

 plt.grid(True)

 plt.subplot(122)

 plt.plot(range(1,epochs), loss[1:], label="Train_loss")

 plt.plot(range(1,epochs), val_loss[1:], label="Test_loss")

 plt.title("Loss over " + str(epochs) + " Epochs", size=15)

 plt.legend()

 plt.grid(True)

 plt.show()

 

plot_acc_loss(result, 100)
test_datagen = ImageDataGenerator(rescale=1. / 255)

eval_generator = test_datagen.flow_from_directory(

 test_dir,target_size=IMAGE_SIZE,

 batch_size=1,

 shuffle=False,

 seed=42,

 

 

 class_mode='categorical')

eval_generator.reset()

x = model.evaluate_generator(eval_generator,

 steps = np.ceil(len(eval_generator) / BATCH_SIZE),

 use_multiprocessing = False,

 verbose = 1,

 workers=1

 )

print("Test loss:" , x[0])

print("Test accuracy:",x[1])