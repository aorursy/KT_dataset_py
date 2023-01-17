# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.optimizers import RMSprop, Adam, SGD, Adadelta
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns
import os

from datetime import datetime
from packaging import version
import tensorflow as tf
from tensorflow import keras

import tensorboard
tensorboard.__version__

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
   # for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%load_ext tensorboard
!rm -rf ./logs/
base_dir = '/kaggle/working/'
train_dr = '/kaggle/input/eye-dataset/Eye dataset/'
test_dr ='/kaggle/input/eye-dataset/Eye dataset/'
pixels = 60
colormode = 'grayscale'
classmode = 'binary'
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

"""Load data with Ternsorflow image generator"""
train_generator = train_datagen.flow_from_directory(train_dr,target_size=(pixels, pixels), color_mode=colormode, class_mode = classmode, batch_size = 20)
test_generator = test_datagen.flow_from_directory(test_dr,target_size=(pixels, pixels), color_mode=colormode, class_mode = classmode, batch_size = 20)
"""Construct the CNN model"""
CNN_Model = Sequential([
    layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu', input_shape=(pixels,pixels,1)),
    layers.MaxPooling2D(pool_size=(2, 2),strides=None,padding='valid'),
    layers.Dropout(0.25),

    layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2),strides=None,padding='valid'),
    layers.Dropout(0.25),

    layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2),strides=None,padding='valid'),
    layers.Dropout(0.25),

    layers.Flatten(),
    
    layers.Dense(400,activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(4,activation='softmax') # Final Layer using Softmax
    
    ])
""" COMPILE The Model """
CNN_Model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',metrics = ['acc'])
"""""" """Define the Keras TensorBoard callback."""
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
""" FIT THE MODEL"""
FITCNN= CNN_Model.fit(train_generator,validation_data = test_generator,epochs = 10,verbose = 1 )
""" SAVE THE MODEL"""
CNN_Model.save(
    base_dir, overwrite=True, include_optimizer=True, save_format='tf',
    signatures=None, options=None
)

CNN_Model.save_weights(
    base_dir, overwrite=True, save_format=None
)

""" CNN MODEL ARCHITECTURE"""
CNN_Model.summary(
    line_length=None, positions=None, print_fn=None
)
""" PLOT THE ACCURACY AND LOSS """
plot_model(CNN_Model,to_file='/kaggle/working/model.png',show_shapes=True,show_layer_names=True, expand_nested=True)

plt.figure(figsize=(24,8))
plt.subplot(1,2,1)
plt.plot(FITCNN.history["val_acc"], label="validation_accuracy", c="red", linewidth=4)
plt.plot(FITCNN.history["acc"], label="training_accuracy", c="green", linewidth=4)
plt.ylabel("Accuracy")
plt.xlabel("Number of Epochs")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(FITCNN.history["val_loss"], label="validation_loss", c="red", linewidth=4)
plt.plot(FITCNN.history["loss"], label="training_loss", c="green", linewidth=4)
plt.ylabel("Loss")
plt.xlabel("Number of Epochs")
plt.legend()
plt.grid(True)

plt.suptitle("ACC / LOSS",fontsize=18)
plt.show()
%tensorboard --logdir logs