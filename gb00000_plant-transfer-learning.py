# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('jpg'):
            continue
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


samp_sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')


#train
#test
#samp_sub

import cv2
import re
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, image
from keras.applications.resnet50 import preprocess_input, decode_predictions, resnet50
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential 
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#get ids in order in the directory
#get train and test file paths
train_filepaths = []
test_filepaths = []
train_ids = []
test_ids = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.startswith('Train'):
            train_ids.append(int(re.findall('\d+' ,filename)[0]))
            train_filepaths.append(os.path.join(dirname, filename))
        if filename.startswith('Test'):
            test_ids.append(int(re.findall('\d+' ,filename)[0]))
            test_filepaths.append(os.path.join(dirname, filename))
len(train_ids), len(test_ids), len(train_filepaths), len(test_filepaths)
def convert_to_tensor(path_img):
    img = image.load_img(path_img, target_size = (224,224))
    img_arr = image.img_to_array(img)
    return np.expand_dims(img_arr, axis = 0)
def convert_all_tensor(paths_imgs):
    tensor_list = [convert_to_tensor(i) for i in paths_imgs]
    return np.vstack(tensor_list)
train_tensors = convert_all_tensor(train_filepaths).astype('float64')/255
test_tensors = convert_all_tensor(test_filepaths).astype('float64')/255
#train_tensors
#test_tensors
train_tensors.shape, test_tensors.shape
train_fileord = train.iloc[train_ids]
#train_fileord
train_y = train_fileord[['healthy', 'multiple_diseases', 'rust', 'scab']].values
#train_y
test_fileord = test.iloc[test_ids]
#test_fileord
X_train, X_test, y_train, y_test = train_test_split(train_tensors, train_y)
train_datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator() 
train_generator = train_datagen.flow(X_train, y_train)
test_generator = test_datagen.flow(X_test, y_test)

#import ssl

#ssl._create_default_https_context = ssl._create_unverified_context
vgg16 = VGG16(weights='imagenet', include_top=False) 
#vgg19 = VGG19(weights='imagenet', include_top=False) 
vgg16.summary()
#vgg19.summary()
x = vgg16.output
#x = vgg19.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
preds = Dense(4, activation='softmax')(x)
#model = Model(inputs=vgg16.input,outputs=preds)
model = Model(inputs=vgg19.input,outputs=preds)
for layer in vgg16.layers:
#for layer in vgg19.layers:
    layer.trainable = False
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])
#model.fit(X_train, y_train,epochs=3,  validation_data=(X_test, y_test), verbose=1)
model.fit_generator(train_generator,
                    epochs=5,
                    validation_data=test_generator, verbose=2)
#for i, layer in enumerate(vgg16.layers):
#   print(i, layer.name)
#for layer in vgg16.layers[:14]:
#   layer.trainable = False
for layer in vgg16.layers:#[14:]:
#for layer in vgg19.layers:
   layer.trainable = True
model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(train_tensors, train_y,epochs=20,  verbose=1)
sub_generator = train_datagen.flow(train_tensors, train_y)
model.fit_generator(sub_generator,
                    epochs=20, verbose=2)
y_pred = model.predict(test_tensors)
y_pred_max = y_pred.copy()


y_pred_idxs = [np.argmax(i) for i in y_pred]
y_pred_idxs
for i in range(len(y_pred_idxs)):
    y_pred_max[i][y_pred_idxs[i]] = 1
    for j in range(4):
        if y_pred_max[i][j] != 1:
            y_pred_max[i][j] = 0
#y_pred_df = pd.DataFrame(y_pred, index = test_ids, columns=['healthy', 'multiple_diseases', 'rust', 'scab'])
y_pred_df_max = pd.DataFrame(y_pred_max, index = test_ids, columns=['healthy', 'multiple_diseases', 'rust', 'scab'], dtype='int64')
sub = samp_sub.copy()
sub[['healthy', 'multiple_diseases', 'rust', 'scab']] = y_pred_df_max.sort_index()
sub.to_csv('submission.csv', index=False)


