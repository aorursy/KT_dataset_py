import os, cv2
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
img_width = 150
img_height = 150

data_dir = '../input/flowers-recognition/flowers/flowers/'

images = []
flower_types = []
for flower_type in os.listdir(data_dir):
    flower_dir = data_dir + flower_type
    flower_files = [flower_dir + '/' + filename for filename in os.listdir(flower_dir)]
    for filename in flower_files:
        if filename.endswith('jpg'):
            images.append(cv2.resize(cv2.imread(filename), (img_width,img_height), interpolation=cv2.INTER_CUBIC))
            flower_types.append(flower_type)
            
images = np.array(images)
flower_types = np.array(flower_types)

le = LabelEncoder()
flower_types_encoded = le.fit_transform(flower_types)
flower_types_encoded_onehot = np_utils.to_categorical(flower_types_encoded)
_, ax = plt.subplots(5,6, figsize=(30,30))
for i in range(5):
    for j in range(6):
      ax[i,j].imshow(cv2.cvtColor(images[(i*1000)+j], cv2.COLOR_BGR2RGB))
      ax[i,j].axis('off')
      ax[i,j].set_title(le.inverse_transform(flower_types_encoded[(i*1000)+j]))
images, flower_types, flower_types_encoded = shuffle(images, flower_types, flower_types_encoded)
from keras.applications.vgg16 import VGG16
weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg16_model = VGG16(include_top=False, weights=weights)
from os import makedirs
from os.path import join, exists, expanduser

cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)
!cp  ../input/vgg16/* ~/.keras/models/
bottleneck_features = vgg16_model.predict(images)
input_shape = bottleneck_features.shape[1:]
num_classes = flower_types_encoded_onehot.shape[1]

model_top = Sequential()
model_top.add(Flatten(input_shape=input_shape))
model_top.add(Dense(256, activation='relu'))
model_top.add(Dense(128, activation='relu'))
model_top.add(Dense(64, activation='relu'))
model_top.add(Dropout(0.5))
model_top.add(Dense(num_classes, activation='sigmoid'))

model_top.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

print (model_top.summary())
history = model_top.fit(bottleneck_features, 
                        flower_types_encoded_onehot,
                        epochs=30,
                        batch_size=16,
                        validation_split=0.3)
print (history.history['val_acc'][-1])
# prediction_values = model_top.predict(bottleneck_features[-1][np.newaxis, :])[0]
# predicted_class = prediction_values.argmax()

_, ax = plt.subplots(5,6, figsize=(30,30))
for i in range(5):
    for j in range(6):
      ax[i,j].imshow(cv2.cvtColor(images[(i*1000)+j], cv2.COLOR_BGR2RGB))
      ax[i,j].axis('off')
        
      prediction_values = model_top.predict(bottleneck_features[(i*1000)+j][np.newaxis, :])[0]
      predicted_class = prediction_values.argmax()

      ax[i,j].set_title(le.inverse_transform(flower_types_encoded[(i*1000)+j]))