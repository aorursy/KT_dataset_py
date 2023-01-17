!pip install keras_vggface
!pip install tensorflow==1.13.2
import tensorflow as tf
print(tf.__version__)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np 
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array, load_img
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import cv2
base_dir = '/kaggle/input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/'

data = []
labels = []

# Walk through all the images and convert them to arrays to be fed into the network

for subdir, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.pkl') is False:
            filepath = subdir + os.sep + file
            image = load_img(filepath, target_size=(224,224))
            # image = cv2.resize(image, (122,122))
            image = img_to_array(image)
            data.append(image)
        
            label = filepath.split(os.path.sep)[-2]
            labels.append(label)
        
        else:
            continue
data = np.stack(data)
data /= 255.0
labels = np.array(labels)

print(np.unique(labels))

# Shuffle the image data and labels in unison 
X = data
y = labels
le=LabelEncoder()
y=le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# Used for new models of tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
training_images = np.expand_dims(X_train, axis=3)
testing_images = np.expand_dims(X_test, axis=3)

train_datagen = ImageDataGenerator(
    
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator()
train_datagen.fit(X_train)
validation_datagen.fit(X_test)
print(training_images.shape)
print(testing_images.shape)
# # Used for old keras models
# model.save("model.h5")
# converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file('model.h5')
# tfmodel = converter.convert()
# open("model.tflite" , "wb") .write(tfmodel)
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop

# Generate the trained model and set all layers to be trainable
trained_model = VGG16(input_shape=(224,224,3), include_top=False)

for layer in trained_model.layers:
    layer.trainable = True

# Construct the model and compile
mod1 = Flatten()
mod_final = Dense(5, activation='softmax')

model = Sequential([trained_model, mod1, mod_final])
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=1e-5), metrics=['accuracy'])
# Fit the model to the data and validate
# Fit the model to the data and validate
history=model.fit_generator(train_datagen.flow(X_train, y_train,
                                     batch_size=32),
                        epochs=30,
                        validation_data=validation_datagen.flow(X_test, y_test,batch_size=32),
                        verbose=2)
# Plot the model results using seaborn and matplotlib
sns.set(style='darkgrid')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
model.save("model.h5")
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file('model.h5')
tfmodel = converter.convert()
open("model.tflite" , "wb") .write(tfmodel)