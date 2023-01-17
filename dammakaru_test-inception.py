# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
TRAIN_DIR = '/kaggle/input/datasetv2/dataset/'

TEST_DIR = '/kaggle/input/datasetv2/dataset/'
import tensorflow as tf

tf.test.gpu_device_name()
from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from keras.applications.inception_v3 import InceptionV3, preprocess_input



CLASSES = 2

    

# setup model

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output

x = GlobalAveragePooling2D(name='avg_pool')(x)

x = Dropout(0.4)(x)

predictions = Dense(CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

   

# transfer learning

for layer in base_model.layers:

    layer.trainable = False
model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator



WIDTH = 299

HEIGHT = 299

BATCH_SIZE = 32



# data prep

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')
validation_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(HEIGHT, WIDTH),

    batch_size=BATCH_SIZE,

    class_mode='categorical')

    

validation_generator = validation_datagen.flow_from_directory(

    TEST_DIR,

    target_size=(HEIGHT, WIDTH),

    batch_size=BATCH_SIZE,

    class_mode='categorical')
import matplotlib.pyplot as plt

x_batch, y_batch = next(train_generator)



plt.figure(figsize=(12, 9))

for k, (img, lbl) in enumerate(zip(x_batch, y_batch)):

    plt.subplot(4, 8, k+1)

    plt.imshow((img + 1) / 2)

    plt.axis('off')
EPOCHS = 20

BATCH_SIZE = 8

STEPS_PER_EPOCH = 80

VALIDATION_STEPS = 30



MODEL_FILE = 'image_classifier_inception.model'



history = model.fit_generator(

    train_generator,

    epochs=EPOCHS,

    steps_per_epoch=STEPS_PER_EPOCH,

    validation_data=validation_generator,

    validation_steps=VALIDATION_STEPS)

  

model.save(MODEL_FILE)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('InceptionV3 model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('InceptionV3 model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# def plot_training(history):

#     acc = history.history['acc']

#     val_acc = history.history['val_acc']

#     loss = history.history['loss']

#     val_loss = history.history['val_loss']

#     epochs = range(len(acc))

  

#     plt.plot(epochs, acc, 'r.')

#     plt.plot(epochs, val_acc, 'r')

#     plt.title('Training and validation accuracy')



#     plt.figure()

#     plt.plot(epochs, loss, 'r.')

#     plt.plot(epochs, val_loss, 'r-')

#     plt.title('Training and validation loss')

#     plt.show()

    

# plot_training(history)
# import numpy as np

# import matplotlib.gridspec as gridspec



# from keras.preprocessing import image





# def predict(model, img):

#     """Run model prediction on image

#     Args:

#         model: keras model

#         img: PIL format image

#     Returns:

#         list of predicted labels and their probabilities 

#     """

#     x = image.img_to_array(img)

#     x = np.expand_dims(x, axis=0)

#     x = preprocess_input(x)

#     preds = model.predict(x)

#     return preds[0]





# def plot_preds(img, preds):

#     """Displays image and the top-n predicted probabilities in a bar graph

#     Args:

#         preds: list of predicted labels and their probabilities

#     """

#     labels = ("code", "notcode")

#     gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

#     plt.figure(figsize=(3,3))

#     plt.subplot(gs[0])

#     plt.imshow(np.asarray(img))

#     plt.subplot(gs[1])

#     plt.barh([0, 1], preds, alpha=0.5)

#     plt.yticks([0, 1], labels)

#     plt.xlabel('Probability')

#     plt.xlim(0, 1)

#     plt.tight_layout()