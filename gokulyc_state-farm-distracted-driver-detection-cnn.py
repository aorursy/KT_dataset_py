# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    if 'test' in dirname:
        print('yes')
        print(dirname)
        break
#     print(dirname)
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
for d1,_,filenames in os.walk('/kaggle/input/state-farm-distracted-driver-detection/imgs/test'):
#     print(d1)
    for filename in filenames:
        print(os.path.join(dirname, filename))
        break
#     print(filenames)
import os
import pandas as pd 
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import pickle
# import tensorflow.compat.v1 as tf
import tensorflow as tf
import tensorflow.keras as keras
# tf.disable_eager_execution()
# def cache_data(data, path):
#     file = open(path, 'wb')
#     pickle.dump(data, file)
#     file.close()
   

# def restore_data(path):
#     data = dict()
#     file = open(path, 'rb')
#     data = pickle.load(file)
#     return data
df_driver_imglist=pd.read_csv("/kaggle/input/state-farm-distracted-driver-detection/driver_imgs_list.csv")
df_driver_imglist.head()
df_driver_imglist.describe()
class_names=sorted(df_driver_imglist['classname'].unique())
class_names
class_i=[class_names.index(i) for i in class_names ]
class_i
df_driver_imglist['classname'].value_counts()
df_driver_imglist['subject'].unique()


# datagen = ImageDataGenerator(rescale=1.0/255)
train_path='/kaggle/input/state-farm-distracted-driver-detection/imgs/train'
test_path='/kaggle/input/state-farm-distracted-driver-detection/imgs/test'
# cache_train='/kaggle/working/state-farm-distracted-driver-detection/cache/train.dat'
# cache_test='/kaggle/working/state-farm-distracted-driver-detection/cache/test.dat'

img_rows, img_cols = 64, 64
def load_data(d_path,start,end):
    x_train=[]
    y_train=[]
    for class_ in class_i[start:end]:
        for dirname, _, filenames in os.walk(d_path):
#     print(dirname)
            for filename in filenames:
                img_data=cv2.imread(os.path.join(dirname, filename))
#                 plt.imshow(img_data)
                print(img_data.shape)
                y_train.append(class_names.index(class_))
                im_conv=cv2.resize(img_data, (img_cols, img_rows))
#                 x_train.append(im_conv)
#                 plt.imshow(cv2.resize(img_data, (img_cols, img_rows)))
#                 print(x_train[0].shape)
                break
    return np.array(x_train,dtype=np.int16),np.array(y_train,dtype=np.int8)
#         df_driver_imglist[]
    
    
load_data(train_path,1,10)
# from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 224
generator=ImageDataGenerator(rescale=1./255) # ,shear_range=0.2,zoom_range=0.2,horizontal_flip=True
train_gen=generator.flow_from_directory(train_path,target_size=(64,64),batch_size=128,class_mode='categorical',shuffle=True)


train_gen
test_gen
# imagegen = ImageDataGenerator()

# train_img =image_dataset_from_directory(train_path,label_mode='int',batch_size=128,validation_split=0.2,subset='training', image_size=(64, 64),seed=32)
# 
# validation_img = image_dataset_from_directory(train_path,label_mode='int',batch_size=128,validation_split=0.2,subset='validation',image_size=(64, 64),seed=32)
# test_img = imagegen.flow_from_directory(test_path, class_mode="categorical",  batch_size=128, target_size=(64, 64))


x_train,y_train=load_data(train_path,0,)
y_train[0:5]
len(x_train)
os.makedirs(os.path.dirname(cache_train), exist_ok=True)
os.makedirs(os.path.dirname(cache_test), exist_ok=True)

cache_data((x_train,y_train),cache_train)

# del (x_train,y_train)
(x_train,y_train)=restore_data(cache_train)
len(x_train),len(y_train)
# 
x_test,y_test=load_data(test_path)
cache_data((x_test,y_test),cache_test)

model=keras.Sequential([#keras.layers.Input(shape=(None,28,28,1)),
                        keras.layers.Conv2D(128,(5,5),padding='same',kernel_initializer='he_uniform',input_shape=(64,64,3)),
                        keras.layers.MaxPooling2D(2),
                        keras.layers.BatchNormalization(),
                        keras.layers.Conv2D(64,(5,5),padding='same'),
                        keras.layers.MaxPooling2D(2),
                        keras.layers.Dropout(0.4),
                        keras.layers.Flatten(),
                        keras.layers.Dense(1024,activation='relu'),
                        keras.layers.Dropout(0.4),
                        keras.layers.Dense(10,activation='softmax'),
                        
                       ])
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
wsave=model.get_weights()
# model.set_weights(wsave)
# x_np=x_train.reshape(-1,64,64,3)
model.fit(train_gen,epochs=20)
# model.fit(train_gen,epochs=20)
# Epoch 1/20
# 176/176 [==============================] - 560s 3s/step - loss: 4.7924 - accuracy: 0.4134
# Epoch 2/20
# 176/176 [==============================] - 566s 3s/step - loss: 0.6814 - accuracy: 0.7651
# Epoch 3/20
# 176/176 [==============================] - 544s 3s/step - loss: 0.3770 - accuracy: 0.8730
# Epoch 4/20
# 176/176 [==============================] - 551s 3s/step - loss: 0.2498 - accuracy: 0.9166
# Epoch 5/20
# 176/176 [==============================] - 537s 3s/step - loss: 0.1995 - accuracy: 0.9344
# Epoch 6/20
# 176/176 [==============================] - 543s 3s/step - loss: 0.1550 - accuracy: 0.9473
# Epoch 7/20
# 176/176 [==============================] - 588s 3s/step - loss: 0.1314 - accuracy: 0.9559
# Epoch 8/20
# 176/176 [==============================] - 591s 3s/step - loss: 0.1127 - accuracy: 0.9637
# Epoch 9/20
# 176/176 [==============================] - 589s 3s/step - loss: 0.0896 - accuracy: 0.9686
# Epoch 10/20
# 176/176 [==============================] - 582s 3s/step - loss: 0.0777 - accuracy: 0.9726
# Epoch 11/20
# 176/176 [==============================] - 583s 3s/step - loss: 0.0734 - accuracy: 0.9750
# Epoch 12/20
# 176/176 [==============================] - 576s 3s/step - loss: 0.0690 - accuracy: 0.9755
# Epoch 13/20
# 176/176 [==============================] - 578s 3s/step - loss: 0.0708 - accuracy: 0.9759
# Epoch 14/20
# 176/176 [==============================] - 569s 3s/step - loss: 0.0825 - accuracy: 0.9719
# Epoch 15/20
# 176/176 [==============================] - 582s 3s/step - loss: 0.0817 - accuracy: 0.9735
# Epoch 16/20
# 176/176 [==============================] - 572s 3s/step - loss: 0.1041 - accuracy: 0.9708
# Epoch 17/20
# 176/176 [==============================] - 575s 3s/step - loss: 0.2228 - accuracy: 0.9351
# Epoch 18/20
# 176/176 [==============================] - 557s 3s/step - loss: 0.1069 - accuracy: 0.9681
# Epoch 19/20
# 176/176 [==============================] - 556s 3s/step - loss: 0.1081 - accuracy: 0.9686
# Epoch 20/20
# 176/176 [==============================] - 556s 3s/step - loss: 0.0943 - accuracy: 0.9755
# model.fit_generator(train_img,epochs=15,use_multiprocessing=True)
# model.fit_generator(train_img,epochs=15,use_multiprocessing=True)
# Epoch 1/15
# 18/18 [==============================] - 20s 1s/step - loss: 1.0998 - accuracy: 0.6480
# Epoch 2/15
# 18/18 [==============================] - 20s 1s/step - loss: 0.9360 - accuracy: 0.7156
# Epoch 3/15
# 18/18 [==============================] - 21s 1s/step - loss: 0.7927 - accuracy: 0.7667
# Epoch 4/15
# 18/18 [==============================] - 20s 1s/step - loss: 0.6767 - accuracy: 0.8063
# Epoch 5/15
# 18/18 [==============================] - 20s 1s/step - loss: 0.5600 - accuracy: 0.8502
# Epoch 6/15
# 18/18 [==============================] - 20s 1s/step - loss: 0.4846 - accuracy: 0.8713
# Epoch 7/15
# 18/18 [==============================] - 20s 1s/step - loss: 0.4197 - accuracy: 0.8926
# Epoch 8/15
# 18/18 [==============================] - 20s 1s/step - loss: 0.3482 - accuracy: 0.9148
# Epoch 9/15
# 18/18 [==============================] - 20s 1s/step - loss: 0.3001 - accuracy: 0.9281
# Epoch 10/15
# 18/18 [==============================] - 20s 1s/step - loss: 0.2664 - accuracy: 0.9386
# Epoch 11/15
# 18/18 [==============================] - 20s 1s/step - loss: 0.2311 - accuracy: 0.9471
# Epoch 12/15
# 18/18 [==============================] - 19s 1s/step - loss: 0.2076 - accuracy: 0.9533
# Epoch 13/15
# 18/18 [==============================] - 20s 1s/step - loss: 0.1824 - accuracy: 0.9585
# Epoch 14/15
# 18/18 [==============================] - 19s 1s/step - loss: 0.1708 - accuracy: 0.9622
# Epoch 15/15
# 18/18 [==============================] - 20s 1s/step - loss: 0.1539 - accuracy: 0.9650
print(model.fit.__doc__)
import shutil, os, glob
srcDir=test_path
dstDir='/kaggle/working/test/img/'

# if os.path.isdir(srcDir) and os.path.isdir(dstDir) :
#         # Iterate over all the files in source directory
#         for filePath in glob.glob(srcDir + '\*'):
#             # Move each file to destination Directory
#             shutil.copy(filePath, dstDir);
# else:
#     print("srcDir & dstDir should be Directories")

shutil.copytree(srcDir,dstDir)

os.listdir(dstDir)

generator_test=ImageDataGenerator(rescale=1./255)
test_gen=generator_test.flow_from_directory('/kaggle/working/test/',target_size=(64,64),batch_size=128,class_mode=None,shuffle=False)
len(test_gen.filenames)
scores=model.evaluate_generator(generator_test)

y_pred
