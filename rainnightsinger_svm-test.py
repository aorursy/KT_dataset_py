import tensorflow as tf

import os

from tensorflow import keras

from tensorflow.keras import layers

import numpy as np

import pathlib

import glob

from PIL import Image

import matplotlib.pyplot as plt

import random

import IPython.display as display

import cv2

from numpy import *

from sklearn import svm

from sklearn.metrics import accuracy_score

%matplotlib inline
print(format(tf.__version__))
tf.test.is_gpu_available()
#tf.config.list_physical_devices('GPU')
os.listdir('../input/thyroid-trans/Picosmos_tran')

data_dir= '../input/thyroid-trans/Picosmos_tran'
data_root=pathlib.Path(data_dir)
for item in data_root.iterdir():

     print(item)
all_image_path = list(data_root.glob('*/*')) 
len(all_image_path)
random.shuffle(all_image_path)
all_image_path[-3:]
all_image_path = [str(path) for path in all_image_path]
images = np.zeros((418, 256, 256))

i = 0
for one_image_path  in all_image_path:

    image = cv2.imread(one_image_path,cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image,(256,256))

    images[i] = image

    i +=1
images.shape
plt.imshow(images[415])
train_images_svm = images.reshape(images.shape[0], -1) 
train_images_svm.shape
test_images_svm = train_images_svm[0:79]
test_images_svm.shape
train_images_svm = train_images_svm[79:]
train_images_svm.shape
image_count=len(all_image_path)
label_names = sorted(item.name for item in data_root.glob('*/')if item.is_dir()) 
label_names
label_to_index =dict((name,index) for index,name in enumerate(label_names))#获取编码
label_to_index
all_image_label = [label_to_index[pathlib.Path(p).parent.name] for p in all_image_path]
all_image_label[200:210]
test_labels_list = all_image_label[0:79]
all_image_label = all_image_label[79:]
all_image_path[:5]
train_images_svm.shape
svc = svm.SVC()

svc.fit(train_images_svm, all_image_label)
y_pred_svm = svc.predict(test_images_svm)

svc_score = accuracy_score(test_labels_list, y_pred_svm)

svc_score