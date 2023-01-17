# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
print('Session Started')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
import zipfile
import gc
import matplotlib.pyplot as plt
%matplotlib inline
filedir = '../input/dogs-vs-cats-redux-kernels-edition/train.zip'
testdir = '../input/dogs-vs-cats-redux-kernels-edition/test.zip'
archive = zipfile.ZipFile(filedir,'r')
archive.extractall()
gc.collect()
def convert_zip_2_np(myfile):
    imglist = []
    label = []
    c = -1
    for dirname, _, filenames in os.walk(myfile):
        for filename in filenames:
            c+=1
            if c<2600:
                img_path = os.path.join(dirname, filename)
                img = cv2.imread(img_path,cv2.IMREAD_COLOR)
                img_class = filename[:3]
                img_resize = cv2.resize(img, (400,300))
    #             img = Image.open(img_path)
    #             img=img.resize((400,300), Image.ANTIALIAS)
                imglist.append(img_resize)
                label.append(img_class)
    return label,np.array(imglist)
labels,x = convert_zip_2_np('train/')
train = pd.DataFrame(labels)
del labels
gc.collect()
train.columns = ['label']
y = train['label'].str.get_dummies()
y = y['cat']
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, Activation


img_rows, img_cols = 300, 400
num_classes = 1
model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(20, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x, y,
          batch_size=20,
          epochs=10,
          validation_split = 0.3)
model.evaluate(x,y)
archive_test = zipfile.ZipFile(testdir,'r')
archive_test.extractall()
gc.collect()
def convert_zip_2_np_test(myfile):
    imglist = []
    ids = []
    for dirname, _, filenames in os.walk(myfile):
        for filename in filenames:
            ids.append(filename[:-4])
            img_path = os.path.join(dirname, filename)
            img = cv2.imread(img_path,cv2.IMREAD_COLOR)
            img_resize = cv2.resize(img, (400,300))
            imglist.append(img_resize)
    return np.array(imglist),ids
test_x,test_id = convert_zip_2_np_test('test/')
prediction = model.predict(test_x)
gc.collect()
del x
prediction[:5]
