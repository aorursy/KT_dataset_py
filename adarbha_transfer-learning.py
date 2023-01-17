# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
from keras.applications import VGG16

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


test_image = load_img('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/NORMAL2-IM-0569-0001.jpeg')
test_image = test_image.resize((150,150))
test_image_arr = img_to_array(test_image)
test_image_arr.shape
plt.imshow(test_image)
filter_lam = lambda x: x != '.DS_Store'
## Training
# Positives
positives_dir_name = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/"
# Normal
normal_dir_name = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/"

train_pos_dir = list(filter(filter_lam, os.listdir(positives_dir_name)))
train_nor_dir = list(filter(filter_lam, os.listdir(normal_dir_name)))

## Validation
# Positivea
val_pos_dir_name = '/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/'
# Normal
val_nor_dir_name = '/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL/'

val_pos_dir = list(filter(filter_lam, os.listdir(val_pos_dir_name)))
val_nor_dir = list(filter(filter_lam, os.listdir(val_nor_dir_name)))

## Test
# Positives
test_pos_dir_name = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/"
# Normal
test_nor_dir_name = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/"

test_pos_dir = list(filter(filter_lam, os.listdir(test_pos_dir_name)))
test_nor_dir = list(filter(filter_lam, os.listdir(test_nor_dir_name)))

#training_size
n_train = len(train_pos_dir) + len(train_nor_dir)

#testing_size
n_test = len(test_pos_dir) + len(test_nor_dir)

#validation_size
n_val = len(val_pos_dir) + len(val_nor_dir)
# Image size set to 150,150 - This is the input size of VGG16
IMAGE_SIZE = 150
# Create dummy matrices 
X_train = np.zeros((n_train,IMAGE_SIZE,IMAGE_SIZE,3))
X_val = np.zeros((n_val, IMAGE_SIZE, IMAGE_SIZE, 3))
X_test = np.zeros((n_test, IMAGE_SIZE, IMAGE_SIZE, 3))


# Rescale the data at this point
X_train = X_train * (1./255)
X_val = X_val * (1./255)
X_test = X_test * (1./255)
# Loop through training images to add it to X_train
for idx,val in enumerate(train_pos_dir):
    X_train[idx,:,:,:] = img_to_array(load_img(os.path.join(positives_dir_name,val)).resize((IMAGE_SIZE,IMAGE_SIZE)))
    
for idx_,val in enumerate(train_nor_dir):
    X_train[(idx+1) + idx_,:,:,:] = img_to_array(load_img(normal_dir_name + val).resize((IMAGE_SIZE,IMAGE_SIZE)))

# Add Y train
Y_train = [1] * len(train_pos_dir)
Y_train.extend([0] * len(train_nor_dir))
Y_train = np.array(Y_train)
# Loop through validation images to add it to X_val
for idx,val in enumerate(val_pos_dir):
    X_val[idx,:,:,:] = img_to_array(load_img(val_pos_dir_name + val).resize((IMAGE_SIZE,IMAGE_SIZE)))
    
for idx_,val in enumerate(val_nor_dir):
    X_val[(idx+1) + idx_,:,:,:] = img_to_array(load_img(val_nor_dir_name + val).resize((IMAGE_SIZE,IMAGE_SIZE)))

# Add Y train
Y_val = [1] * len(val_pos_dir)
Y_val.extend([0] * len(val_nor_dir))
Y_val = np.array(Y_val)
# Loop through validation images to add it to X_test
for idx,val in enumerate(test_pos_dir):
    X_test[idx,:,:,:] = img_to_array(load_img(test_pos_dir_name + val).resize((IMAGE_SIZE,IMAGE_SIZE)))
    
for idx_,val in enumerate(test_nor_dir):
    X_test[(idx+1) + idx_,:,:,:] = img_to_array(load_img(test_nor_dir_name + val).resize((IMAGE_SIZE,IMAGE_SIZE)))

# Add Y train
Y_test = [1] * len(test_pos_dir)
Y_test.extend([0] * len(test_nor_dir))
Y_test = np.array(Y_test)
print(f"Train label freq: {pd.Series(Y_train).value_counts()}")
print(f"Validatin label freq: {pd.Series(Y_val).value_counts()}")
print(f"Test label freq: {pd.Series(Y_test).value_counts()}")
import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
from keras.optimizers import RMSprop, adam

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))


model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))


for layer in model.layers[0:4]:
    layer.trainable = False

model.compile(loss = 'binary_crossentropy',optimizer = RMSprop(), metrics=[tf.keras.metrics.Recall(), 'acc'])
model.summary()
hist = None
with tf.device("/device:GPU:0"):
    hist = model.fit(
    x = X_train, y = Y_train,
    validation_data = (X_val, Y_val),
    shuffle=True,
    batch_size=64,
    epochs=30,
    verbose=1)
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(30)

plt.figure()
plt.plot(epochs, loss, label = 'loss')
plt.plot(epochs, val_loss, label = 'val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.figure()
plt.plot(epochs, acc, label = 'acc')
plt.plot(epochs, val_acc, label = 'val_acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
score = model.evaluate(X_test, Y_test)
score
X_pred = model.predict(X_test).flatten()
predictor = lambda x:0 if x < 0.5 else 1
x_pred_ = list(map(predictor, X_pred))
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
print('Acc: {}'.format(accuracy_score(y_true = Y_test, y_pred = x_pred_)))
print('Recall: {}'.format(recall_score(y_true = Y_test, y_pred = x_pred_)))
print('Precision: {}'.format(precision_score(y_true = Y_test, y_pred = x_pred_)))
print('f1: {}'.format(f1_score(y_true = Y_test, y_pred = x_pred_)))
df = pd.DataFrame({'y_actual':Y_test, 'y_predicted':x_pred_}, columns=['y_actual','y_predicted'])
confusion_matrix = pd.crosstab(df['y_actual'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True)
base_model.summary()
# block5_conv3 - unfreeze this layer - to train some inner layer parameters
base_model.trainable = True
for layer in base_model.layers:
    if layer.name == "block5_conv3":
        layer.trainable = True
    else:
        layer.trainable = False
base_model.summary()
# Redo the whole model with new trainable weights

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',optimizer = RMSprop(), metrics=[tf.keras.metrics.Recall(), 'acc'])

hist = None
with tf.device("/device:GPU:0"):
    hist = model.fit(
    x = X_train, y = Y_train,
    validation_data = (X_val, Y_val),
    shuffle=True,
    batch_size=64,
    epochs=30,
    verbose=1)
    

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(30)

plt.figure()
plt.plot(epochs, loss, label = 'loss')
plt.plot(epochs, val_loss, label = 'val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.figure()
plt.plot(epochs, acc, label = 'acc')
plt.plot(epochs, val_acc, label = 'val_acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
X_pred = model.predict(X_test).flatten()
predictor = lambda x:0 if x < 0.5 else 1
x_pred_ = list(map(predictor, X_pred))

print('Acc: {}'.format(accuracy_score(y_true = Y_test, y_pred = x_pred_)))
print('Recall: {}'.format(recall_score(y_true = Y_test, y_pred = x_pred_)))
print('Precision: {}'.format(precision_score(y_true = Y_test, y_pred = x_pred_)))
print('f1: {}'.format(f1_score(y_true = Y_test, y_pred = x_pred_)))
df = pd.DataFrame({'y_actual':Y_test, 'y_predicted':x_pred_}, columns=['y_actual','y_predicted'])
confusion_matrix = pd.crosstab(df['y_actual'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True)
