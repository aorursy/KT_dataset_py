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
import os

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
np.random.seed(20)
from sklearn.metrics import accuracy_score

import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras import backend as K
from keras.optimizers import Adam  
from keras.callbacks import ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
benign_train = '../input/skin-cancer-malignant-vs-benign/train/benign'
malignant_train = '../input/skin-cancer-malignant-vs-benign/train/malignant'

benign_test = '../input/skin-cancer-malignant-vs-benign/test/benign'
malignant_test = '../input/skin-cancer-malignant-vs-benign/test/malignant'
read = lambda image_name: np.asarray(Image.open(image_name).convert("RGB"))

# load training images
images_benign = [read(os.path.join(benign_train, filename)) for filename in os.listdir(benign_train)]
X_benign = np.array(images_benign, dtype='uint8')

images_malignant = [read(os.path.join(malignant_train, filename)) for filename in os.listdir(malignant_train)]
X_malignant = np.array(images_malignant, dtype='uint8')

# load testing images
images_benign = [read(os.path.join(benign_train, filename)) for filename in os.listdir(benign_train)]
X_benign_test = np.array(images_benign, dtype='uint8')

images_malignant = [read(os.path.join(malignant_test, filename)) for filename in os.listdir(malignant_test)]
X_malignant_test = np.array(images_malignant, dtype='uint8')
# create labels 
y_benign = np.zeros(X_benign.shape[0])
y_malignant = np.ones(X_malignant.shape[0])

y_benign_test = np.zeros(X_benign_test.shape[0])
y_malignant_test = np.ones(X_malignant_test.shape[0])
# merge data
X_train = np.concatenate((X_benign, X_malignant), axis = 0)
y_train = np.concatenate((y_benign, y_malignant), axis = 0)

X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)
y_test = np.concatenate((y_benign_test, y_malignant_test), axis = 0)
# shuffle data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
y_train = y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
y_test = y_test[s]
# display few images of moles to see how they are classified

# image dimensions for display
w = 30
h = 10
fig = plt.figure(figsize=(12,8))
columns = 5
rows = 2

for i in range(1, columns*rows + 1):
    ax = fig.add_subplot(rows, columns, i)
    if( y_train[i] == 0 ):
        ax.title.set_text('Benign sample')
    else:
        ax.title.set_text('Malignant sample')
    plt.imshow(X_train[i], interpolation = 'nearest')
plt.show()
# Convert labels into one hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
# data Normalization
X_train = X_train/255
X_test = X_test/255
# ResNet-50 Model building
input_shape = (224,224,3)
lr = 1e-5 # learning rate
epochs = 50
batch_size = 32

# set learning rate reduction
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',
                                           patience=5,
                                           verbose=1,
                                           factor=0.5,
                                           min_lr=1e-7)
model = ResNet50(include_top = True,
                weights = None,
                input_tensor = None, 
                input_shape=input_shape,
                pooling='avg',
                classes=2)
model.compile(optimizer = Adam(lr),
             loss = 'binary_crossentropy',
             metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.2,
                   epochs = 20,
                   batch_size = batch_size,
                   verbose = 1,
                   callbacks = [learning_rate_reduction])
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('ResNet50 model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ResNet50 model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
y_pred = model.predict(X_test)
print(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

# save model
# serialize model to JSON
resnet50_json = model.to_json()

with open("resnet50.json", "w") as json_file:
    json_file.write(resnet50_json)
    
# serialize weights to HDF5
model.save_weights("resnet50.h5")
print("Saved model to disk")