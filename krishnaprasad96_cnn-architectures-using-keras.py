# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import keras
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16
base_dir = '/kaggle/input/intel-image-classification/'
training_dir = base_dir + 'seg_train/seg_train/'
testing_dir = base_dir + 'seg_test/seg_test/'

print("No of Images for class Buildings",len(os.listdir(training_dir+'buildings')))
print("No of Images for class glacier",len(os.listdir(training_dir+'glacier')))
print("No of Images for class sea",len(os.listdir(training_dir+'sea')))
print("No of Images for class mountain",len(os.listdir(training_dir+'mountain')))
print("No of Images for class forest",len(os.listdir(training_dir+'forest')))
print("No of Images for class street",len(os.listdir(training_dir+'street')))

train_gen = ImageDataGenerator(
    rescale = 1.0/255.0,
    zoom_range = 0.2,
    shear_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    fill_mode = 'nearest',
    horizontal_flip=True    
)

test_gen = ImageDataGenerator(
    rescale = 1.0/255.0
)

training_data = train_gen.flow_from_directory(
    training_dir,
    batch_size=32,
    class_mode = 'categorical',
    color_mode ='rgb',
    target_size=(150, 150)
)

testing_data = test_gen.flow_from_directory(
    testing_dir,
    batch_size=32,
    class_mode = 'categorical',
    color_mode ='rgb',
    target_size=(150, 150)
)
class cnn_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if(logs['accuracy']>0.5):
            print("Training Stopped")
            self.model.stop_training = True
callback = cnn_callback()

def visualization(history):
    acc = history.history['acc']
    loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']
    
    fig, ax = plt.subplots()
    ax.plot(acc, label='Training Accuracy')
    ax.plot(val_acc, label='Validation Accuracy')
    leg = ax.legend();
    plt.show()
    
    fig, ax = plt.subplots()
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    leg = ax.legend();
    plt.show()

def save_weights(model, name):
    base_dir = "/kaggle/working/"
    model_dir = base_dir + name
    
    # serialize weights to HDF5
    model.save_weights(model_dir+".h5")
    print("Saved model to disk")
    
def load_weights(model, name):
    base_dir = "/kaggle/working/"
    model_dir = base_dir + name
    
    # load weights into new model
    model.load_weights(model_dir + ".h5")
    print("Loaded weights from disk")
    return model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, (5,5), input_shape=(150, 150, 3), activation='tanh'),
    tf.keras.layers.AveragePooling2D(2,2),
    tf.keras.layers.Conv2D(16, (5,5), activation='tanh'),
    tf.keras.layers.AveragePooling2D(2,2),
    tf.keras.layers.Conv2D(120, (1,1), activation='tanh'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(84, activation='tanh'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.summary()
#Training the model
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
history = model.fit(training_data, epochs = 15, validation_data = testing_data, callbacks=[callback])
visualization(history)
save_weights(model, 'lenet_5')
loaded_model = load_weights(model, 'lenet_5')
vgg_16_basedir = '/kaggle/input/vgg16/'
vgg16_model = VGG16(input_shape=(150, 150, 3), include_top=False, weights = vgg_16_basedir+'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
for layer in vgg16_model.layers:
    layer.trainable = False

#Checking if the Dense layers were removed.
print(vgg16_model.layers[-1])
model = vgg16_model.output
model = keras.layers.core.Flatten()(model)
model = keras.layers.core.Dense(1024, activation='relu')(model)
model = keras.layers.core.Dropout(0.2)(model)
model = keras.layers.core.Dense(512, activation='relu')(model)
model = keras.layers.core.Dense(6, activation='softmax')(model)

final_model = keras.models.Model(inputs = vgg16_model.input, outputs = model)
final_model.summary()
final_model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['acc'])
history = final_model.fit_generator(training_data, epochs = 15, validation_data = testing_data, callbacks=[callback])
visualization(history)
model_input = keras.layers.Input(shape=(150, 150, 3))
conv_1 = keras.layers.Conv2D(64, (1,1), activation='relu', padding='same')(model_input)
conv_3 = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(model_input)
conv_5 = keras.layers.Conv2D(32, (5,5), activation='relu', padding='same')(model_input)
max_pooling = keras.layers.MaxPooling2D((2, 2), padding='same')(model_input)
layer_1 = keras.layers.merge([conv_1, conv_2, conv_3, max_pooling])