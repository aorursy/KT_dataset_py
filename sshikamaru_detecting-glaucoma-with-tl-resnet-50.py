

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_label = pd.read_csv('/kaggle/input/glaucoma-detection/glaucoma.csv')

y_train = train_label['Glaucoma']

train_label.head()
from numpy import asarray



from PIL import Image

# load the image

image = Image.open('/kaggle/input/glaucoma-detection/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Validation/Glaucoma_Positive/613.jpg')

# summarize some details about the image

print(image.format)

print(image.mode)

print(image.size)

# show the image

image.show()

pixels = asarray(image)


# global centering



# calculate global mean

mean = pixels.mean()

print('Mean: %.3f' % mean)

print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

# global centering of pixels

pixels = pixels - mean

# confirm it had the desired effect

mean = pixels.mean()

print('Mean: %.3f' % mean)

print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

print(pixels)





# example of pixel normalization

# confirm pixel range is 0-255

print('Data Type: %s' % pixels.dtype)

print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

# convert from integers to floats

pixels = pixels.astype('float32')

# normalize to the range 0-1

pixels /= 255.0

mean = pixels.mean()

print('pixel mean = ', mean)



# confirm the normalization

print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

print(pixels)
import matplotlib.pyplot as plt

fig, (ax0, ax1) = plt.subplots(1, 2)

ax0.imshow(image)

ax0.axis('off')

ax0.set_title('image')

ax1.imshow(pixels)

ax1.axis('off')

ax1.set_title('result')

plt.show()
from skimage import io



def imshow(image_RGB):

  io.imshow(image_RGB)

  io.show()
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.resnet50 import preprocess_input
TRAIN_DIR = '/kaggle/input/glaucoma-detection/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Train'



TEST_DIR = '/kaggle/input/glaucoma-detection/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Validation'
'''import os

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.preprocessing.image import ImageDataGenerator







model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='softmax'))







train_generator = ImageDataGenerator(

  data_format="channels_last",

  rescale = 1. / 255

)



train_batches = train_generator.flow_from_directory(

    batch_size=32,

    directory='./dataset/train',

    target_size=(256, 256),

    class_mode='binary'

)



validation_generator = ImageDataGenerator(

  data_format="channels_last",

  rescale = 1. / 255

)



validation_batches = validation_generator.flow_from_directory(

    batch_size=32,

    directory='./dataset/validation',

    target_size=(256, 256),

    class_mode='binary'

)



model = create_model()



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



# Starts training the model

model.fit_generator(train_batches,

                    epochs=15,

                    verbose=1,

                    steps_per_epoch=len(train_batches),

                    validation_data=validation_batches,

                    initial_epoch=0,

                    validation_steps=len(validation_batches)

                    )



test_generator = ImageDataGenerator(

    data_format='channels_last',

    rescale=1./255

)



test_batches = test_generator.flow_from_directory(

    batch_size=1,

    directory='./dataset/test',

    target_size=[256, 256],

    class_mode='binary'

)



score = model.evaluate_generator(test_batches, verbose=1)



print(model.metrics_names)

print('test dataset: ' + str(score))'''
from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout

from keras.models import Sequential, Model 

from keras.optimizers import SGD, Adam

from keras.callbacks import TensorBoard

import keras

import matplotlib.pyplot as plt



HEIGHT = 300

WIDTH = 300



BATCH_SIZE = 8

class_list = ["class_1", "class_2"]

FC_LAYERS = [1024, 512, 256]

dropout = 0.5

NUM_EPOCHS = 100

BATCH_SIZE = 8



def build_model(base_model, dropout, fc_layers, num_classes):

    for layer in base_model.layers:

        layer.trainable = False



    x = base_model.output

    x = Flatten()(x)

    for fc in fc_layers:

        print(fc)

        x = Dense(fc, activation='relu')(x)

        x = Dropout(dropout)(x)

    preditions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs = base_model.input, outputs = preditions)

    return finetune_model



base_model_1 = ResNet50(weights = 'imagenet',

                       include_top = False,

                       input_shape = (HEIGHT, WIDTH, 3))



train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,

                                   rotation_range = 90,

                                   horizontal_flip = True,

                                   vertical_flip = True,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   zoom_range=0.1,)



test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,

                                  rotation_range = 90,

                                  horizontal_flip = True,

                                  vertical_flip = False)



train_generator = train_datagen.flow_from_directory(TRAIN_DIR,

                                                    target_size = (HEIGHT, WIDTH),

                                                    batch_size = BATCH_SIZE)



test_generator = test_datagen.flow_from_directory(TEST_DIR,

                                                  target_size = (HEIGHT, WIDTH),

                                                  batch_size = BATCH_SIZE)









resnet50_model = build_model(base_model_1,

                                      dropout = dropout,

                                      fc_layers = FC_LAYERS,

                                      num_classes = len(class_list))



adam = Adam(lr = 0.00001)

resnet50_model.compile(adam, loss="binary_crossentropy", metrics=["accuracy"])



filepath = "./checkpoints" + "RestNet50" + "_model_weights.h5"

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor = ["acc"], verbose= 1, mode = "max")

cb=TensorBoard(log_dir=("/home/ubuntu/"))

callbacks_list = [checkpoint, cb]



print(train_generator.class_indices)



resnet50_model.summary()
history = resnet50_model.fit_generator(generator = train_generator, epochs = NUM_EPOCHS, steps_per_epoch = 100, 

                                       shuffle = True, validation_data = test_generator)
import matplotlib.pyplot as plt

%matplotlib inline

image_batch,label_batch = train_generator.next()



print(len(image_batch))

for i in range(0,len(image_batch)):

    image = image_batch[i]

    print(label_batch[i])

    imshow(image)
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.optimizers import RMSprop
base_model_2 = InceptionV3(weights = 'imagenet',

                       include_top = False,

                       input_shape = (HEIGHT, WIDTH, 3))
inception_model = build_model(base_model_2,

                                      dropout = dropout,

                                      fc_layers = FC_LAYERS,

                                      num_classes = len(class_list))

inception_model.compile(optimizer = RMSprop(lr = 0.00001), loss="binary_crossentropy", metrics=["accuracy"])

inception_model.summary()
history_2 = inception_model.fit_generator(generator = train_generator, epochs = NUM_EPOCHS, steps_per_epoch = 100, 

                                       shuffle = True, validation_data = test_generator)