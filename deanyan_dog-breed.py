# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import cv2
import random
path_root = '../input'
path_train = os.path.join(path_root, 'train')
path_test = os.path.join(path_root, 'test')
path_labels_csv = os.path.join(path_root, 'labels.csv')
path_result_csv = os.path.join(path_root, 'sample_submission.csv')

len_train = len(os.listdir(path_train))
len_test = len(os.listdir(path_test))
labels = pd.read_csv(path_labels_csv)
print(labels.head(5))

breed = list(set(labels['breed']))
n_classes = len(breed)
print(n_classes, len(os.listdir(path_train)))

labels.shape
class_to_num = dict(zip(breed, range(n_classes)))
num_to_class = dict(zip(range(n_classes), breed))
width = 300
X_train = np.zeros((len_train, width, width, 3), dtype = np.uint8)
y_train = np.zeros((len_train, n_classes))
n = len(labels)

for i in tqdm(range(n)):
    X_train[i] = cv2.resize(cv2.imread(os.path.join(path_train, '%s.jpg' % labels['id'][i])), dsize = (width, width))
    y_train[i][class_to_num[labels['breed'][i]]] = 1

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

plt.figure(figsize = (15, 9))
for i in range(12):
    index = random.randint(0, len_train - 1)
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_train[index][:, :, ::-1])
    plt.title(num_to_class[y_train[index].argmax()])
from keras.models import *
from keras.layers import *
from keras.applications import resnet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
resnet_model = resnet50.ResNet50(include_top = False, weights='imagenet', input_shape = (300, 300, 3))
    
inputs = Input((width, width, 3))
x = inputs
x = resnet_model(x)
x = GlobalAveragePooling2D()(x)
resnet_model = Model(inputs, x)

resnet_model.summary()

features = resnet_model.predict(X_train, batch_size=64, verbose=1)
inputs = Input(features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(n_classes, activation='softmax')(x)
model = Model(inputs, x)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

h = model.fit(features, y_train, batch_size=128, epochs=10, validation_split=0.1)
import matplotlib.pyplot as plt

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')

plt.subplot(1, 2, 2)
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.ylabel('acc')
plt.xlabel('epoch')
test_data = pd.read_csv(path_result_csv)

n_test = len(test_data)
X_test = np.zeros((len_test, width, width, 3), dtype=np.uint8)
for i in tqdm(range(len_test)):
    X_test[i] = cv2.resize(cv2.imread(os.path.join(path_test, '%s.jpg' % test_data['id'][i])), (width, width))
features_test = resnet_model.predict(X_test, batch_size=64, verbose=1)
y_pred = model.predict(features_test, batch_size=128)

for b in breed:
    test_data[b] = y_pred[:,class_to_num[b]]
    
test_data.to_csv('pred.csv', index=None)
