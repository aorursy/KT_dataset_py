
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
# Import library 
import numpy as np
import pandas as pd 
import matplotlib.pylab as plt
import tensorflow as tf
import scipy.io as sio
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
import keras
import h5py
# Dataset
training_data = sio.loadmat('/kaggle/input/svhndataset/train_32x32.mat')
testing_data = sio.loadmat('/kaggle/input/svhndataset/test_32x32.mat')
# Load images and labels

X_tr, y_tr = training_data['X'], training_data['y']
X_te, y_te = testing_data['X'], testing_data['y']
print(X_tr.shape)
print(y_tr.shape)
print(X_te.shape)
print(y_te.shape)
# Fix the axes of the images

X_tr = np.moveaxis(X_tr, -1, 0)
X_te = np.moveaxis(X_te, -1, 0)

print(X_tr.shape)
print(X_te.shape)
# Normalize the images data

print('Min: {}, Max: {}'.format(X_tr.min(), X_te.max()))
# Convert train and test images into 'float32' type

X_tr = X_tr.astype('float32') / 255
X_te = X_te.astype('float32') / 255

print(X_tr.shape)
y_tr[y_tr == 10] = 0
y_te[y_te == 10] = 0
print(np.unique(y_tr))
from sklearn.preprocessing import OneHotEncoder
 
# Fit the OneHotEncoder
enc = OneHotEncoder().fit(y_tr.reshape(-1, 1))

# Transform the label values to a one-hot-encoding.
y_tr = enc.transform(y_tr.reshape(-1, 1)).toarray()
y_te = enc.transform(y_te.reshape(-1, 1)).toarray()

print("Training set", y_tr.shape)
print("Test set", y_te.shape)
print(y_tr.shape)
print(y_te.shape)
print(X_tr[0].shape)
from keras import backend as K

def rec(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def pre(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = pre(y_true, y_pred)
    recall = rec(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# Data augmentation

datagen = ImageDataGenerator(rotation_range=8,
                             zoom_range=[0.95, 1.05],
                             height_shift_range=0.10,
                             shear_range=0.15)
# Define actual model

keras.backend.clear_session()

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same', 
                           activation='relu',
                           input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), padding='same', 
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(64, (3, 3), padding='same', 
                           activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(128, (3, 3), padding='same', 
                           activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),    
    keras.layers.Dense(10,  activation='softmax')
])

early_stopping = keras.callbacks.EarlyStopping(patience=8)
optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)
model_checkpoint = keras.callbacks.ModelCheckpoint(
                   '/kaggle/working/best_cnn.h5', 
                   save_best_only=True)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['acc',f1,pre, rec])
#it shows the architecture of the model
model.summary()
# data Augumentation 

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

history = model.fit_generator(datagen.flow(X_tr, y_tr, batch_size=128),
                              epochs=30, validation_data=(X_te, y_te),
                              callbacks=[early_stopping, model_checkpoint],
                              )
#Here we plot the performance of the model on the training and validation data.
plt.plot(history.history["acc"])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Performance')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(["Train Accuracy","Validation Accuracy","Train loss","Validation Loss"])
plt.show()
import matplotlib.pyplot as plt
plt.plot(history.history["acc"])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()
#Here we calculate the accuracy of the model
score = model.evaluate(X_te, y_te, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
loss, accuracy, f1_score, precision, recall = model.evaluate(X_te, y_te, verbose=0)
print(loss)
print(accuracy)
print(f1_score)
print(precision)
print(recall)
res_test = model.predict(X_te)
res_test = pd.DataFrame({'true':np.argmax(y_te, axis=1), 'guess':np.argmax(res_test, axis=1), 'trust':np.max(res_test, axis=1)})
res_test.head(10)
errors = res_test[res_test.true != res_test.guess].sort_values('trust', ascending=False)
errors.head(10)
print('Percentage of error %4.2f %%' % (100 * len(errors)/len(X_te)))
i = 17000
res = model.predict(X_te[i][None,:,:])
print("Image", i)
print(f"Model says it is a {np.argmax(res)} while it is a {np.argmax(y_te[i])}")
print("Stats are", np.array(res))
plt.imshow(X_te[i])
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='Validation')
pyplot.legend()
pyplot.show()
# Get convolutional layers

layers = [model.get_layer('conv2d_1'), 
          model.get_layer('conv2d_2'),
          model.get_layer('conv2d_3'),
          model.get_layer('conv2d_4'),
          model.get_layer('conv2d_5')]
# Define a model which gives the outputs of the layers

layer_outputs = [layer.output for layer in layers]
activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
# Create a list with the names of the layers

layer_names = []
for layer in layers:
    layer_names.append(layer.name)
    
print(layer_names)
# Model with unseen data and prediction 
img = image.load_img('/kaggle/input/testsvhn/testsvhn/test80.png', target_size=(224,224))
img
import cv2 as cv
from matplotlib import pyplot as plt
import pytesseract
from keras.preprocessing import image

img_path = '/kaggle/input/testsvhn/testsvhn/test80.png'
img = image.load_img(img_path, target_size=(32, 32))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
print('Input image shape:', x.shape)
preds = model.predict(x)
print(preds)
print("Predicted Value: ", np.argmax(preds))

i = 16000 #1318
res = model.predict(X_te[i][None,:,:])
print(res)