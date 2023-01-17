# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
os.getcwd()
files = list()
for dirname, _, filenames in os.walk('/kaggle/input/'):
     for filename in filenames:
         files.append(os.path.join(dirname, filename))
files[0:5]
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#!pip install lime shap
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import lime
import shap
import os
import matplotlib.pyplot as plt
import cv2
import random
def create_stack(lst):
    temp_list = list()
    for file in lst:
        img = cv2.imread(file)
        img = cv2.resize(img, (150,150))
        temp_list.append(img)
    return np.array(temp_list)
# files = random.sample(files, len(files))
# files[0:5]
files_train = list()
for dirname, _, filenames in os.walk('/kaggle/input/messy-vs-clean-room/images/train/clean/'):
     for filename in filenames:
         files_train.append(os.path.join(dirname, filename))
for dirname, _, filenames in os.walk('/kaggle/input/messy-vs-clean-room/images/train/messy/'):
     for filename in filenames:
         files_train.append(os.path.join(dirname, filename))
# print(files_train[:5])
files_train = random.sample(files_train, len(files_train))
# print(files_train[:5])
x_train = create_stack(files_train)
#x_train.shape
files_test = list()
for dirname, _, filenames in os.walk('/kaggle/input/messy-vs-clean-room/images/test/'):
     for filename in filenames:
         files_test.append(os.path.join(dirname, filename))
#files_test
x_test = create_stack(files_test)
#x_test.shape
plt.imshow(x_train[0])
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_dir = '/kaggle/input/messy-vs-clean-room/images/train'

train_generator = train_datagen.flow_from_directory(train_dir, target_size = (150,150),
                                                   batch_size = 10,
#                                                    color_mode = 'grayscale',
                                                   class_mode = 'binary')
validation_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
val_dir = '/kaggle/input/messy-vs-clean-room/images/val'

val_generator = validation_datagen.flow_from_directory(val_dir, 
                                    target_size = (150,150),
                                    batch_size = 10,
#                                     color_mode = 'grayscale',
                                    class_mode = 'binary',
                                    shuffle=False)
def RoomModel(input_shape):
  X_input = Input(input_shape)
  X = X_input
    
  #Layer 1 - Conv>BN>RELU>MaxPool
  
  X = Conv2D(256,(5,5),strides = (1,1), name = 'conv0', padding = 'same')(X)
  X = BatchNormalization(axis = 3, name = 'bn0')(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2),strides = (2,2), name = 'maxpool0')(X)
  
  #Layer 2 - Conv>BN>RELU>MaxPool
  
  X = Conv2D(128,(5,5),strides = (1,1), name = 'conv1', padding = 'same')(X)
  X = BatchNormalization(axis = 3, name = 'bn1')(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2),strides = (2,2), name = 'maxpool1')(X)
  X = Dropout(rate = 0.25)(X)
  
  #Layer 3 - Conv>BN>RELU>MaxPool
  
  X = Conv2D(64,(3,3),strides = (1,1), name = 'conv2', padding = 'valid')(X)
  X = BatchNormalization(axis = 3, name = 'bn2')(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2),strides = (2,2), name = 'maxpool2')(X)
  X = Dropout(rate = 0.25)(X)
  
  #Layer 4 - Conv>BN>RELU>MaxPool

  X = Conv2D(32,(3,3),strides = (1,1), name = 'conv3', padding = 'same')(X)
  X = BatchNormalization(axis = 3, name = 'bn3')(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2),strides = (2,2), name = 'maxpool3')(X)
  X = Dropout(rate = 0.25)(X)

  #Layer 5 - Conv>BN>RELU>MaxPool

  X = Conv2D(16,(3,3),strides = (1,1), name = 'conv4', padding = 'same')(X)
  X = BatchNormalization(axis = 3, name = 'bn4')(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2),strides = (2,2), name = 'maxpool4')(X)
  X = Dropout(rate = 0.25)(X)
  
  #FC layers with flattening

  X = Flatten()(X)
  X = Dense(128, activation = 'relu', name = 'fc1')(X)
  X = Dropout(rate = 0.25)(X)
  X = Dense(32, activation = 'relu', name = 'fc2')(X)
  X = Dropout(rate = 0.25)(X)
  X = Dense(1, activation = 'sigmoid', name = 'preds')(X)
  
  #model instance creator
  model = Model(inputs = X_input, outputs = X, name = 'RoomModel')
  model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  
  return model
input_shape = [150,150,3]
roomModel = RoomModel(input_shape)
my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
history = roomModel.fit(train_generator,
                              epochs=100,
                              validation_data = val_generator,
                              verbose=1,
                              callbacks = my_callbacks)
%matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')
from sklearn.metrics import classification_report, confusion_matrix

roomModel.load_weights('/kaggle/working/model.84-0.34.h5')

Y_pred = roomModel.predict(val_generator, verbose = 0)
print(Y_pred)
y_pred = (Y_pred>0.5) 
print(y_pred)
class_labels = {v: k for k, v in val_generator.class_indices.items()}

print('Confusion Matrix')
print(confusion_matrix(val_generator.classes, y_pred))

print('Classification Report')
target_names = list(class_labels.values())
print(classification_report(val_generator.classes, y_pred, target_names=target_names))
from lime import lime_image
import time
def predict_fn(image):
#     for image in images:
#         if(image.shape[2]==3):
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         image = cv2.resize(image, (150,150))
    image = ((image.astype(float))/255)
    image = image.reshape((-1,150,150,3))
    return roomModel.predict(image)
explainer = lime_image.LimeImageExplainer()
plt.imshow(x_test[0])
predict_fn(x_test[0])
explanation_0 = explainer.explain_instance(x_test[0], predict_fn, top_labels=5, hide_color=0, num_samples=1000)
from skimage.segmentation import mark_boundaries

temp, mask = explanation_0.get_image_and_mask(explanation_0.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp, mask))
ind =  explanation_0.top_labels[0]

#Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation_0.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation_0.segments) 

#Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
# plt.imshow(mark_boundaries(temp, mask))