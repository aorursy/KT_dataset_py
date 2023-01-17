import numpy as np # linear algebra

import random

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import tensorflow as tf



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras import optimizers

from tensorflow.keras import backend as K

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.callbacks import ModelCheckpoint

import PIL

import matplotlib.pyplot as plt

import json

from IPython.display import Image as disp_image 

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import seaborn as sns

import matplotlib.pyplot as plt

from PIL import Image

import cv2

import shutil

from glob import glob

# Helper libraries

import matplotlib.pyplot as plt

import math

%matplotlib inline

print(tf.__version__)
#数据路径

data_root='/kaggle/input/covidct/'

path_positive_cases = os.path.join('/kaggle/input/covidct/CT_COVID/')

path_negative_cases = os.path.join('/kaggle/input/covidct/CT_NonCOVID/')
# jpg and png files

positive_images_ls = glob(os.path.join(path_positive_cases,"*.png"))



negative_images_ls = glob(os.path.join(path_negative_cases,"*.png"))

negative_images_ls.extend(glob(os.path.join(path_negative_cases,"*.jpg")))
covid = {'class': 'CT_COVID',

         'path': path_positive_cases,

         'images': positive_images_ls}



non_covid = {'class': 'CT_NonCOVID',

             'path': path_negative_cases,

             'images': negative_images_ls}
# Create Train-Test Directory

subdirs  = ['train/', 'test/']

for subdir in subdirs:

    labeldirs = ['CT_COVID', 'CT_NonCOVID']

    for labldir in labeldirs:

        newdir = subdir + labldir

        os.makedirs(newdir, exist_ok=True)
#划分训练集和测试集

random.seed(237)

#测试集比例

test_ratio = 0.2





for cases in [covid, non_covid]:

    total_cases = len(cases['images']) #全部图象数

    num_to_select = int(test_ratio * total_cases) #测试集属灵

    

    print(cases['class'], num_to_select)

    

    list_of_random_files = random.sample(cases['images'], num_to_select) 



    for files in list_of_random_files:

        shutil.copy2(files, 'test/' + cases['class'])
# 建立训练集和测试集文件

for cases in [covid, non_covid]:

    image_test_files = os.listdir('test/' + cases['class'])

    for images in cases['images']:

        if images.split('/')[-1] not in (image_test_files): 

            shutil.copy2(images, 'train/' + cases['class'])
#使用VGG的图像调整，增强图像

img_height = 224

img_width = 224

channels = 3

batch_size = 32

epochs = 200



train_datagen = ImageDataGenerator(rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   rescale=1./255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   #horizontal_flip=True,

                                   fill_mode='nearest',

                                   validation_split=0.2)



test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory(

    './train',

     target_size = (img_height, img_width),

     batch_size = batch_size,

     class_mode = 'binary',

     subset = 'training',

     shuffle=True)



validation_set = train_datagen.flow_from_directory(

    './train',

     target_size = (img_height, img_width),

     batch_size = batch_size,

     class_mode = 'binary',

     subset = 'validation',

     shuffle=False)



test_set = train_datagen.flow_from_directory(

    './test',

     target_size = (img_height, img_width),

     batch_size = 1,

     shuffle = False,

     class_mode = 'binary')



print(training_set.class_indices)
# 图像示例

img_files = os.listdir('train/CT_NonCOVID')

img_path = img_files[np.random.randint(0,len(img_files))]



img = cv2.imread('train/CT_NonCOVID/{}'.format(img_path))

#img.thumbnail((500, 500))

#display(img)
# VGG16 导入权重和网络

model = VGG16(weights='../input/vgg16weight/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape = (img_width, img_height, channels))

for layer in model.layers[:-5]:

    layer.trainable = False

#除去全连接网络层

#设计网络

top_model = Sequential()

top_model.add(model)

top_model.add(Flatten())

top_model.add(Dense(256, activation='relu'))

top_model.add(Dropout(0.5))

top_model.add(Dense(1, activation='sigmoid'))



print(model.summary())

print(top_model.summary())
# 训练网络

top_model.compile(loss='binary_crossentropy',

                  optimizer=optimizers.RMSprop(lr=1e-4, decay=1e-6),

                  metrics=['accuracy'])



history = top_model.fit_generator(

          training_set,

          steps_per_epoch=training_set.n // batch_size,

          epochs=epochs,

          validation_data=validation_set,

          validation_steps=validation_set.n // batch_size)



# 保存网络

top_model.save('covid_model.h5', save_format='h5')
# 画图显示

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

 

epochs = range(len(acc))

 

plt.plot(epochs, acc, 'b', label='Training Accuracy')

plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')

plt.title('Training and validation accuracy')

plt.legend()

 

plt.figure()

 

plt.plot(epochs, loss, 'b', label='Training Loss')

plt.plot(epochs, val_loss, 'r', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()

 

plt.show()
# 测试

test_pred = top_model.evaluate_generator(test_set,

                                        steps=test_set.n//batch_size,

                                        use_multiprocessing=False,

                                        verbose=1)



print('Test loss: ', test_pred[0])

print('Test accuracy: ', test_pred[1])
test_pred = top_model.predict(test_set,

                              steps=test_set.n,

                              use_multiprocessing=False,

                              verbose=1)



test_pred_class = (test_pred >= 0.5)*1



# 混淆矩阵

cm = confusion_matrix(test_set.classes,

                      test_pred_class)



ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells



ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(['COVID-19', 'Non_COVID-19']); ax.yaxis.set_ticklabels(['COVID-19', 'Non_COVID-19']);