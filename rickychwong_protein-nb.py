import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import gc
from keras.preprocessing.image import ImageDataGenerator
#================================
# import the necessary packages
#from keras.models import Sequential
#from keras.layers.normalization import BatchNormalization
#from keras.layers.convolutional import Conv2D
#from keras.layers.convolutional import MaxPooling2D
#from keras.layers.core import Activation
#from keras.layers.core import Flatten
#from keras.layers.core import Dropout
#from keras.layers.core import Dense
#from keras import backend as K

#================================

import matplotlib
#matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
#from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
#from imutils import paths
import numpy as np
#import argparse
import random
import pickle
import cv2
import os

from PIL import Image
from collections import Counter

from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.resnet50 import ResNet50 as PTModel, preprocess_input

from tensorflow.python.keras.applications import ResNet50, InceptionResNetV2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D,BatchNormalization, MaxPooling2D
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.layers.core import Activation, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.tree
import sklearn.ensemble

from tqdm import tqdm

from os.path import join, exists, expanduser
cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    os.makedirs(models_dir)
nepochs = 36
#nsamples = 1200
img_size=256
nclass=28
train = pd.read_csv("../input/human-protein-atlas-image-classification/train.csv")
train.head()
train['Labels'] = train['Target'].map(lambda x: [int(y) for y in x.split(' ')])
for i in range(nclass):
    train[i] = train['Labels'].map(lambda x: int(i in x))
train.head()
counts = train[list(range(nclass))].sum().sort_values(ascending=False)
counts
temp = pd.DataFrame()

for i in range(nclass):
    l = min(counts[i],50)
    if i == 0:
        temp = train[train[i]>0][:l]
    else:
        temp = temp.append(train[train[i]>0][:l]).drop_duplicates(subset='Id')
train = temp
counts = train[list(range(nclass))].sum().sort_values(ascending=False)
counts
train.head()
def read_img(train_test,img_id,size):
    img_r = Image.open('../input/human-protein-atlas-image-classification/'+train_test+'/'+img_id+'_red.png').resize((size,size))
    img_g = Image.open('../input/human-protein-atlas-image-classification/'+train_test+'/'+img_id+'_green.png').resize((size,size))
    img_b = Image.open('../input/human-protein-atlas-image-classification/'+train_test+'/'+img_id+'_blue.png').resize((size,size))
    return preprocess_input(np.expand_dims(np.stack([img_r,img_g,img_b],-1).copy(), axis=0))
x_trains = np.zeros((len(train), img_size, img_size, 3), dtype='float32')
for i, img_id in tqdm(enumerate(train['Id'])):
    x_trains[i]  = read_img('train',img_id, img_size)
y_trains = train[list(range(nclass))].values 
!ls ../input/keras-pretrained-models/
if False:
    resnet_weights_path = '../input/keras-pretrained-models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
    os.system("cp ../input/keras-pretrained-models/inception_resnet_v2* ~/.keras/models/")
    model = Sequential()
    model.add(InceptionResNetV2(include_top=False, pooling='avg', 
                                weights=resnet_weights_path, input_shape = (img_size, img_size, 3)))
#    for i in range(20):
#        model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
#        model.add(BatchNormalization(axis=-1))
#    model.add(GlobalAveragePooling2D())
#    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
#    model.add(Dense(128, activation='relu'))
#    model.add(Dropout(0.25))
#    model.add(Dense(64, activation='relu'))
    model.add(Dense(nclass, activation='softmax'))
    init_lr=0.001
    opt = Adam(lr=init_lr, decay=init_lr / nepochs)

    # Say not to train first layer (ResNet) model. It is already trained
    model.layers[0].trainable = False
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
if True:
    resnet_weights_path = '../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    os.system("cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/")
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path, input_shape = (img_size, img_size, 3)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
#    model.add(Dense(128, activation='relu'))
#    model.add(Dropout(0.25))
#    model.add(Dense(64, activation='relu'))
    model.add(Dense(nclass, activation='softmax'))
    init_lr=0.001
    opt = Adam(lr=init_lr, decay=init_lr / nepochs)

    # Say not to train first layer (ResNet) model. It is already trained
    model.layers[0].trainable = False
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
batch_size=32
cv_num=10
kfold=sklearn.model_selection.KFold(cv_num,shuffle=True,random_state=42)
histories=[]
if True:
    datagen = ImageDataGenerator(
        samplewise_std_normalization=False,
        rotation_range=360,
        width_shift_range=0,
        height_shift_range=0,
        zoom_range = 0.2,
        featurewise_center = False,
        samplewise_center = False,
        horizontal_flip=True,
        vertical_flip=True)
    
    for i,(train_index,valid_index) in enumerate(kfold.split(y_trains)):
        #x_train = x_trains[train_index]
        #y_train = y_trains[train_index]
        #x_valid = x_trains[valid_index]
        #y_valid = y_trains[valid_index]
        
# fits the model on batches with real-time data augmentation:
        histories.append(model.fit_generator(datagen.flow(x_trains[train_index], y_trains[train_index], batch_size=batch_size),
                    steps_per_epoch=len(x_trains[train_index]) // batch_size, 
                    validation_data=datagen.flow(x_trains[valid_index], y_trains[valid_index], batch_size=batch_size),
                    validation_steps=len(y_trains[valid_index])//batch_size,
   #                 callbacks=[EarlyStopping()],
                    epochs=nepochs))
fig, arr = plt.subplots(cv_num,2,figsize=(8,40), sharex=True, sharey='col')
for i in range(cv_num):
    arr[i][0].plot(histories[i].history['acc'])
    arr[i][0].plot(histories[i].history['val_acc'])
    arr[i][0].set_title(str(i)+' accuracy')
    arr[i][0].legend(['train','test'],loc='upper left')

    arr[i][1].plot(histories[i].history['loss'])
    arr[i][1].plot(histories[i].history['val_loss'])
    arr[i][1].set_title(str(i)+' loss')
    arr[i][1].legend(['train','test'],loc='upper left')

plt.show()
submit = pd.read_csv('../input/human-protein-atlas-image-classification/sample_submission.csv')

y_preds_labels = []
for i, img_id in tqdm(enumerate(submit['Id'])):
    y_pred = model.predict(read_img('test',img_id, img_size))
    c = np.arange(nclass)[y_pred[0] >= 0.5]
    label = ' '.join(str(cc) for cc in c)
    y_preds_labels.append(label)

submit['Predicted'] = y_preds_labels
submit.to_csv('submission.csv', index=False)