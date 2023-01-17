import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import inception_resnet_v2

from keras.layers.core import Dropout

from keras.layers.core import Flatten

from keras.layers.core import Dense

from keras.layers import Input

from keras.models import Model

from keras.optimizers import SGD

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import numpy as np

import pickle

import os

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import densenet

from keras.models import Sequential, Model, load_model

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

from keras import regularizers

from keras import backend as K

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

from PIL import Image

import seaborn as sns

%load_ext tensorboard



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
K.set_learning_phase(1)

#https://www.kaggle.com/jutrera/training-a-densenet-for-the-stanford-car-dataset

img_width, img_height = 299,299

nb_train_samples = 1631

nb_validation_samples = 500

epochs = 10

batch_size = 32

n_classes = 4
train_data_dir = '/kaggle/input/disaster/data/train'

validation_data_dir = '/kaggle/input/disaster/data/test'



train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    fill_mode = 'constant',

    cval = 1,

    rotation_range = 5,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='categorical')
def build_model():

    base_model = inception_resnet_v2.InceptionResNetV2(input_shape=(img_width, img_height, 3),

                                     weights='/kaggle/input/full-keras-pretrained-no-top/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',

                                     include_top=False,

                                     pooling='max')

    for layer in base_model.layers:

      layer.trainable = True



    x = base_model.output

    print(x.shape)

    x=Dense(1000)(x)

    x = Dense(500,activation='relu',kernel_constraint= tf.keras.constraints.max_norm(3))(x)

    x = Dense(250,activation='relu',kernel_constraint= tf.keras.constraints.max_norm(3))(x)

    x = Dense(125,activation='relu',kernel_constraint= tf.keras.constraints.max_norm(3))(x)

    x = Dense(60,activation='relu',kernel_constraint= tf.keras.constraints.max_norm(3))(x)

    predictions = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

   

    return model


model = build_model()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

print(model)
#early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta=1e-4)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)

callbacks_list = [reduce_lr]
#add kernel_constraint=maxnorm(3)

epochs=100

#40 is done before

model_history = model.fit_generator(

    train_generator,

    epochs=epochs,

    validation_data=validation_generator,

    steps_per_epoch= nb_train_samples//batch_size,

    validation_steps=nb_validation_samples // batch_size,

    callbacks=callbacks_list)
plt.figure(0)

plt.plot(model_history.history['acc'],'r')

plt.plot(model_history.history['val_acc'],'g')

plt.xticks(np.arange(0, 100, 1.0))

plt.rcParams['figure.figsize'] = (8, 6)

plt.xlabel("Num of Epochs")

plt.ylabel("Accuracy")

plt.title("Training Accuracy vs Validation Accuracy")

plt.legend(['train','validation'])

 

plt.figure(1)

plt.plot(model_history.history['loss'],'r')

plt.plot(model_history.history['val_loss'],'g')

plt.xticks(np.arange(0, 100, 1.0))

plt.rcParams['figure.figsize'] = (8, 6)

plt.xlabel("Num of Epochs")

plt.ylabel("Loss")

plt.title("Training Loss vs Validation Loss")

plt.legend(['train','validation'])

plt.show()
model.evaluate_generator(validation_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)


pred = model.predict_generator(validation_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

predicted = np.argmax(pred, axis=1)
print('Confusion Matrix')

cm = confusion_matrix(validation_generator.classes, np.argmax(pred, axis=1))

plt.figure(figsize = (30,20))

sns.set(font_scale=1.4) #for label size

sns.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size

plt.show()

print("***********************************************************")

print('Classification Report')

#print(classification_report(validation_generator.classes, predicted, target_names=class_names))
print("confusion matrix")

print(cm)
model.save("inception_resnet_v2_model_version_2.h5")