# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import datetime
import time
import re
from tqdm import tqdm


print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# path to training dataset
train_path = '../input/train/'
labels = pd.read_csv('../input/labels.csv')

model_dir = '../input/'
list_images = [train_path+f for f in os.listdir(train_path) if re.search('jpg|JPG', f)]

print(list_images[0:4])
train_labels = os.listdir(train_path)

labels.head(5)

n = len(labels)
breed = set(labels['breed'])
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
n_class = len(breed)

print(n)
print(n_class)

yy = pd.value_counts(labels['breed'])
print(yy[0:5])
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
fig.set_size_inches(20,10)
sns.set_style("whitegrid")

ax = sns.barplot(x = yy.index, y = yy, data = labels)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 10)
ax.set(xlabel='Dog Breed', ylabel='Count')
ax.set_title('Distribution of the Dog Breeds')
targets_series = pd.Series(labels['breed'])
one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)

import cv2
width = 224
orig_label = []
X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)
orig_label = []
for i in tqdm(range(n)):
    X[i] = cv2.resize(cv2.imread('../input/train/%s.jpg' % labels['id'][i]), (width, width))
    y[i] = one_hot_labels[i]
    orig_label.append(labels['breed'][i])

print(orig_label[0:5])
print("Number of Samples:",X.shape[0])
print(y.shape)
num_class = y.shape[1]
print("Number of training lables:",num_class)
import random
import matplotlib.pyplot as plt

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

plt.figure(figsize=(12, 6))
for i in range(8):
    random_index = random.randint(0, n-1)
    plt.subplot(2, 4, i+1)
    plt.imshow(X[random_index][:,:,::-1])
    plt.title(orig_label[random_index])
    plt.axis('off')
df_test = pd.read_csv('../input/sample_submission.csv')
n_test = len(df_test)
X_test = np.zeros((n_test, width, width, 3), dtype=np.uint8)
for i in tqdm(range(n_test)):
    X_test[i] = cv2.resize(cv2.imread('../input/test/%s.jpg' % df_test['id'][i]), (width, width))

print(len(X_test))
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *

def get_features(MODEL, data=X):
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=64, verbose=1)
    return features
vgg16_features = get_features(VGG16, X)
inputs = Input(vgg16_features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax')(x)
model = Model(inputs, x)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(vgg16_features, y, batch_size=128, epochs=10, validation_split=0.1)
vgg16_feature_test = get_features(VGG16, X_test)
y_pred = model.predict(vgg16_feature_test, batch_size=128)
for b in breed:
    df_test[b] = y_pred[:,class_to_num[b]]
 
df_test.to_csv('submission.csv', index=None)