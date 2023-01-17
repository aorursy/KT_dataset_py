# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import glob

import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
class_names = ['cat','dog']

generator = ImageDataGenerator(rescale = 1./255)

train_set = generator.flow_from_directory("../input/dogs-cats-images/dataset/training_set",

                                          class_mode = "binary",

                                          batch_size = 32,

                                          target_size = (224,224))



val_set = generator.flow_from_directory("../input/dogs-cats-images/dataset/test_set",

                                        batch_size = 32,

                                        class_mode = "binary",

                                        target_size = (224,224))
plt.figure(figsize = (20,6))

for i in range(5):

    plt.subplot(1,5,i+1)

    plt.imshow(train_set[0][0][i])

plt.show()
num_classes = 2

epochs = 5



URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(URL,

                                  input_shape = (224,224,3))





feature_extractor.trainable = False

#define model

model = Sequential([

    feature_extractor, #pre-trained model

    tf.keras.layers.Dense(num_classes,activation='softmax') #output model

])



model.summary()
model.compile(optimizer = 'adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics = ['accuracy']

             )
history = model.fit(train_set,validation_data = val_set,epochs = epochs)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure(figsize=(20,6))

plt.subplot(1,2,1)

plt.plot(range(epochs),acc,label = "Training Accuracy")

plt.plot(range(epochs),val_acc,label = "Validation Accuracy")

plt.legend()

plt.title("Training and Validation Accuracy")



plt.subplot(1,2,2)

plt.plot(range(epochs),loss,label = "Training Loss")

plt.plot(range(epochs),val_loss,label="Validation Loss")

plt.legend()

plt.title("Training and Validation Loss")

plt.show()
predictions = [np.argmax(i) for i in model.predict(val_set[0])]

#predictions = [class_names[np.argmax(i)] for i in model.predict(val_set[0])]

plt.figure(figsize = (20,6))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.subplots_adjust(hspace = 0.3)

    plt.imshow(val_set[0][0][i])

    if(predictions[i] == int(val_set[0][1][i])):

        plt.title(class_names[predictions[i]],color='g')

    else:

        plt.title(class_names[predictions[i]],color='r')

    plt.axis('off')

plt.show()