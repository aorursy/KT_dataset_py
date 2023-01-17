import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import json
import math
import cv2
import PIL
from PIL import Image
import seaborn as sns
sns.set(style='darkgrid')
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import layers
from keras.applications import ResNet50,MobileNet, DenseNet201, InceptionV3, NASNetLarge, InceptionResNetV2, NASNetMobile
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K
import gc
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import itertools

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from sklearn.decomposition import PCA

%matplotlib inline
'''
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
'''
sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
import os
print(os.listdir("../input/siim-isic-melanoma-classification"))
#Loading Train and Test Data
train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
test = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
print("{} images in train set.".format(train.shape[0]))
print("{} images in test set.".format(test.shape[0]))
train.head()
test.head()
np.mean(train.target)
plt.figure(figsize=(10,5))
sns.countplot(x='target', data=train,
                   order=list(train['target'].value_counts().sort_index().index) ,
                   color='cyan')
train['target'].value_counts()
train.columns
z=train.groupby(['target','sex'])['benign_malignant'].count().to_frame().reset_index()
z.style.background_gradient(cmap='Reds')  
sns.catplot(x='target',y='benign_malignant', hue='sex',data=z,kind='bar')
images= train['image_name'].values

#extract 9 random images
import random
random_images = [np.random.choice(images + '.jpg') for i in range(9)]

#location of image dir 
image_dir = '../input/siim-isic-melanoma-classification/jpeg/train'

print('Display random images')

#iterate and plot images
for i in range(9):
    plt.subplot(3,3,i+1)
    img = plt.imread(os.path.join(image_dir, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()   
benign = train[train['benign_malignant']=='benign']
malignant = train[train['benign_malignant']=='malignant']
images= benign['image_name'].values

#extract 9 random images
import random
random_images = [np.random.choice(images + '.jpg') for i in range(9)]

#location of image dir 
image_dir = '../input/siim-isic-melanoma-classification/jpeg/train'

print('Display Benign images')

#iterate and plot images
for i in range(9):
    plt.subplot(3,3,i+1)
    img = plt.imread(os.path.join(image_dir, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()   
images= malignant['image_name'].values

#extract 9 random images
import random
random_images = [np.random.choice(images + '.jpg') for i in range(9)]

#location of image dir 
image_dir = '../input/siim-isic-melanoma-classification/jpeg/train'

print('Display Malignant images')

#iterate and plot images
for i in range(9):
    plt.subplot(3,3,i+1)
    img = plt.imread(os.path.join(image_dir, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout() 
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Convolution2D,Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras import applications
import time 
start=time.time()
train_images = np.load('../input/siimisic-melanoma-resized-images/x_train_96.npy')
end=time.time()
print(f"\nTime to load train images: {round(end-start,5)} seconds.")
print('Train_images shape: ',train_images.shape)
start=time.time()
test_images = np.load('../input/siimisic-melanoma-resized-images/x_test_96.npy')
end=time.time()
print(f"\nTime to load test images: {round(end-start,5)} seconds.")
print('Test_images shape: ',test_images.shape)
#target data
train_labels =np.array(train.drop(['image_name', 'patient_id', 'sex', 'age_approx',
       'anatom_site_general_challenge', 'diagnosis','benign_malignant'],axis=1))
print('Train_labels shape: ',train_labels.shape)
#spliting train data
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(train_images,train_labels,test_size=0.3)
print('x_train shape: ',x_train.shape)
print('x_val shape: ',x_val.shape)
augs = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

augs.fit(x_train)
#annealer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
strategy = tf.distribute.get_strategy()
#VGG-16 MODEL NO. 1
from keras.applications.vgg16 import VGG16

input_shape=(96,96,3)
num_classes=1
tmodel_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
model = Sequential()
model.add(tmodel_base)
model.add(BatchNormalization())
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='sigmoid', name='output_layer'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
#XCEPTION MODEL NO. 2
from keras.layers import Dropout, DepthwiseConv2D, MaxPooling2D, concatenate
from keras.models import Model

inp = Input(shape = (96,96, 3))
x = inp
x = Conv2D(32, (3, 3), strides = 2, padding = "same", activation = "relu")(x)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x = Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu")(x)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)

x1 = DepthwiseConv2D((3, 3), (1, 1), padding = "same", activation = "relu")(x)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x1 = DepthwiseConv2D((3, 3), (1, 1), padding = "same", activation = "relu")(x1)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x1 = MaxPooling2D((2, 2), strides = 1)(x1)

x = concatenate([x1, Conv2D(64, (2, 2), strides = 1)(x)])

x1 = Activation("relu")(x)
x1 = Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu")(x1)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x1 = DepthwiseConv2D((3, 3), strides = 1, padding = "same", activation = "relu")(x1)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x1 = DepthwiseConv2D((3, 3), strides = 1, padding = "same")(x1)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x1 = MaxPooling2D((2, 2), strides = 1)(x1)

x = concatenate([x1, Conv2D(256, (2, 2), strides = 1)(x)])


x = Activation("relu")(x)
x = Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu")(x)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x = Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu")(x)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x = Flatten()(x)

x = Dense(1, activation = "sigmoid")(x)


model2 = Model(inp, x)
model2.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model2.summary()
#DENSENET MODEL NO. 1
from tensorflow.keras.applications import DenseNet201
import tensorflow.keras.layers as L

with strategy.scope():
    dnet201 = DenseNet201(
        input_shape=(96,96, 3),
        weights='imagenet',
        include_top=False
    )
    dnet201.trainable = True

    model3 = tf.keras.Sequential([
        dnet201,
        L.GlobalAveragePooling2D(),
        L.Dense(1, activation='sigmoid')
    ])
    model3.compile(
        optimizer='adam',
        loss = 'binary_crossentropy',
        metrics=['accuracy']
    )

model3.summary()
batch_size=128
epochs=30

history = model.fit(x_train,
             y_train,
             batch_size=batch_size,
             nb_epoch=epochs,
             verbose=1,
             validation_data=(x_val,y_val))
batch_size=128
epochs=15

history3 = model2.fit(x_train,
             y_train,
             batch_size=batch_size,
             nb_epoch=epochs,
             verbose=1,
             validation_data=(x_val,y_val))
batch_size=128
epochs=30

history3 = model3.fit(x_train,
             y_train, 
             batch_size=batch_size,
             nb_epoch=epochs,
             verbose=1,
             validation_data=(x_val,y_val))
model.save("vgg16.h5")
model2.save("xception.h5")
model3.save("densenet.h5") 
scores = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
scores = model2.evaluate(x_val, y_val, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
scores = model3.evaluate(x_val, y_val, verbose=0)
print('Test loss_3:', scores[0])
print('Test accuracy_3:', scores[1])
y_test_prob = model.predict(test_images)
pred_df = pd.DataFrame({'image_name': test['image_name'], 'target': np.concatenate(y_test_prob)})
pred_df.to_csv('submission_vgg.csv',header=True, index=False)
pred_df.head(10)
y_test_prob2 = model2.predict(test_images)
pred_df2 = pd.DataFrame({'image_name': test['image_name'], 'target': np.concatenate(y_test_prob2)})
pred_df2.to_csv('submission_xception.csv',header=True, index=False)
pred_df2.head(10)
y_test_prob3 = model3.predict(test_images)
pred_df3 = pd.DataFrame({'image_name': test['image_name'], 'target': np.concatenate(y_test_prob3)})
pred_df3.to_csv('submission_dense.csv',header=True, index=False)
pred_df3.head(10)
en = pd.DataFrame({'image_name':test['image_name'], 'target':(0.3*pred_df['target'] + 0.3*pred_df2['target'] + 0.3*pred_df3['target'])})
en.to_csv('ensemble1.csv',header=True, index=False)
en.head(10)


