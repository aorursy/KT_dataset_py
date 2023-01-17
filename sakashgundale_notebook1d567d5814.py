# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

#import cv2

import tensorflow as tf

from PIL import Image

import os

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import load_img,img_to_array



from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input

from keras.layers import AveragePooling2D

from keras.applications import VGG16

from keras.optimizers import Adam

from keras.models import Model,Sequential

from keras.layers import Input
train_gen=ImageDataGenerator(rescale=1./255)

train_dir='../input/grapes-images/grapes images/train'

train=train_gen.flow_from_directory(train_dir,batch_size=32,class_mode='categorical',target_size=(256,256))
train.class_indices
labels = '\n'.join(sorted(train.class_indices.keys()))



with open('labels.txt', 'w') as f:

  f.write(labels)
val_gen=ImageDataGenerator(rescale=1./255)

val_dir='../input/grapes-images/grapes images/validation'

validation=val_gen.flow_from_directory(val_dir,batch_size=32,class_mode='categorical',target_size=(256,256))
import keras

base_model = keras.applications.VGG16(weights = "imagenet" ,

             include_top=False,input_shape=(256, 256, 3))
base_model.trainable = False



# Create new model on top

inputs = keras.Input(shape=(256, 256, 3))



# The base model contains batchnorm layers. We want to keep them in inference mode

# when we unfreeze the base model for fine-tuning, so we make sure that the

# base_model is running in inference mode here.

x = base_model(inputs)

x = keras.layers.GlobalAveragePooling2D()(x)

x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout

outputs = keras.layers.Dense(4)(x)



vgg16_model = keras.Model(inputs, outputs, name='pretrained_vgg16')

vgg16_model.summary()
vgg16_model.compile(optimizer=keras.optimizers.Adam(),

              loss=keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=[keras.metrics.CategoricalAccuracy()]

)



epochs = 100



# train_num = train_set.samples  # num of training samples

# valid_num = valid_set.samples  # num of validation samples



vgg16_history = vgg16_model.fit_generator(train,

                                steps_per_epoch=150,  # use 150 random batches (= 4800 samples) for training

                                validation_data=validation,

                                epochs=epochs,

                                validation_steps=100  # use 100 random batches (= 3200 samples) for validation 

)
plt.plot(vgg16_history.history['categorical_accuracy'])

plt.plot(vgg16_history.history['val_categorical_accuracy'])

plt.title("VGG-16's Accuracy")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(loc='best')

plt.show()





plt.plot(vgg16_history.history['loss'])

plt.plot(vgg16_history.history['val_loss'])

plt.title("VGG-16's loss")

plt.xlabel('Epochs')

plt.ylabel('loss')

plt.legend(loc='best')

plt.show()

vgg16_model.save('GrapeModel.h5')
pred=load_img('../input/grapes-images/grapes images/train/Grape___Black_rot/003d09ef-e16c-4e8a-badf-847d46cb3dc0___FAM_B.Rot 3184_flipLR.JPG')

plt.imshow(pred)

pred=img_to_array(pred)



pred=pred*(1./255)

pred=pred.reshape(1,256,256,3)

prediction=vgg16_model.predict(pred)

print(prediction)

dicti=train.class_indices

dicti=list(dicti.keys())

print(dicti)

print('Final result is :')

dicti[np.argmax(prediction)]



base_model = keras.applications.MobileNet(

    weights="imagenet",  # load weights pretrained on the ImageNet

    input_shape=(256, 256, 3),

    include_top=False  # do not include the ImageNet classifier at the top

)  
# Freeze the base_model

base_model.trainable = False



# Create new model on top

inputs = keras.Input(shape=(256, 256, 3))



# The base model contains batchnorm layers. We want to keep them in inference mode

# when we unfreeze the base model for fine-tuning, so we make sure that the

# base_model is running in inference mode here.

x = base_model(inputs, training=False)

x = keras.layers.GlobalAveragePooling2D()(x)

x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout

outputs = keras.layers.Dense(4)(x)



mobilenet_model = keras.Model(inputs, outputs, name='pretrained_mobilenet')

mobilenet_model.summary()
mobilenet_model.compile(optimizer=keras.optimizers.Adam(),

                        loss=keras.losses.CategoricalCrossentropy(from_logits=True),

                        metrics=[keras.metrics.CategoricalAccuracy()]

)



epochs = 25



mobilenet_history = mobilenet_model.fit(train,

                                        steps_per_epoch=150, # use 150 random batches (= 4800 samples) for training

                                        

                                        validation_data=validation,

                                        epochs=epochs,

                                        validation_steps=100,  # use 100 random batches (= 3200 samples) for validation 

)
results = mobilenet_model.evaluate(validation)

print('val loss:', results[0])

print('val acc:', results[1])
plt.plot(mobilenet_history.history['categorical_accuracy'])

plt.plot(mobilenet_history.history['val_categorical_accuracy'])

plt.title("MobileNet's Accuracy")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(loc='best')

plt.show()
plt.plot(mobilenet_history.history['loss'])

plt.plot(mobilenet_history.history['val_loss'])

plt.title("MobileNet's Loss")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(loc='best')
pred=load_img('../input/grapes-images/grapes images/train/Grape___Black_rot/003d09ef-e16c-4e8a-badf-847d46cb3dc0___FAM_B.Rot 3184_flipLR.JPG')

plt.imshow(pred)

pred=img_to_array(pred)



pred=pred*(1./255)

pred=pred.reshape(1,256,256,3)

prediction=mobilenet_model.predict(pred)

print(prediction)

dicti=train.class_indices

dicti=list(dicti.keys())

print(dicti)

print('Final result is :')

dicti[np.argmax(prediction)]

base_model.trainable = True

mobilenet_model.summary()
mobilenet_model.compile(optimizer=keras.optimizers.Adam(1e-5),  # set a small learning rate

                        loss=keras.losses.CategoricalCrossentropy(from_logits=True),

                        metrics=[keras.metrics.CategoricalAccuracy()]

)



epochs = 5



mobilenet_ft_history = mobilenet_model.fit(train,

                                           steps_per_epoch=150,   # use 150 random batches (= 4800 samples) for training

                                           validation_data=validation,

                                           epochs=epochs,

                                           validation_steps=100  # use 100 random batches (= 3200 samples) for validation

)
mobilenet_model.save('mobilenet-grapes.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(mobilenet_model)

tflite_model = converter.convert()



# Save the TF Lite model.

with tf.io.gfile.GFile('model.tflite', 'wb') as f:

  f.write(tflite_model)
from tensorflow.keras.models import load_model