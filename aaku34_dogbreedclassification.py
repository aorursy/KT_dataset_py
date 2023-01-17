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

from keras.preprocessing.image import ImageDataGenerator



from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Conv2D, MaxPooling2D,Dense, Flatten, GlobalAveragePooling2D, Activation, Flatten, Dropout, BatchNormalization

from tensorflow.python.keras.applications.resnet50 import preprocess_input



from keras.applications.resnet50 import ResNet50

from keras.applications import xception

from keras.applications import inception_v3

from os.path import join, exists, expanduser

from os import listdir, makedirs

import os

from tqdm import tqdm

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import cv2

import keras as k
data_dir = '../input'

labels = pd.read_csv(join(data_dir, 'labels.csv'))

Test_submission=pd.read_csv(join(data_dir, 'sample_submission.csv'))
Test_submission.head(5)
Test_submission.shape
labels.head(5)
labels.breed.head(5)
labels.pivot_table(index='breed',aggfunc=len).sort_values('id',ascending=False)
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 14))

labels['breed'].value_counts().plot(kind='bar')

plt.show()
im_size = 224



x_train = []

y_train = []



i = 0 

for f, breed in tqdm(labels.values):

    img = cv2.imread('../input/train/{}.jpg'.format(f))

    label = labels.breed[i]

    x_train.append(cv2.resize(img, (im_size, im_size)))

    y_train.append(label)

    i += 1
x_test = []



for f in tqdm(Test_submission['id'].values):

    img = cv2.imread('../input/test/{}.jpg'.format(f))

    x_test.append(cv2.resize(img, (im_size, im_size)))
x_train_new=np.array(x_train)



x_test_new=np.array(x_test)
x_train_new.shape,x_test_new.shape
num_class=120

le=preprocessing.LabelEncoder()

Y=le.fit_transform(y_train)

Y=k.utils.to_categorical(Y, num_class)
Y.shape
pretrained_model = ResNet50(weights='imagenet',include_top=False, pooling='avg')
pretrained_model.layers.pop()
for layer in pretrained_model.layers:

    layer.trainable = False
model = k.models.Sequential()

model.add(pretrained_model)

model.add(k.layers.Dense(512, activation='relu'))

model.add(k.layers.Dropout(.5))

model.add(k.layers.Dense(num_class, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])

callbacks_list = [k.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]

model.summary()

print('Input Shape = {}'.format(model.layers[0].input_shape))

print('Shape Shape = {}'.format(model.layers[-1].output_shape))
model_log=model.fit(

    x=x_train_new, y=Y, batch_size=32,

    epochs=25, verbose=1,validation_split=.2

     

)
fig = plt.figure()

plt.subplot(2,1,1)

plt.plot(model_log.history['acc'])

#plt.plot(model_log.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower right')
prediction=model.predict(x_test_new, verbose=1)
sub = pd.DataFrame(prediction)
Test_submission1=Test_submission
sub.insert(0, 'id', Test_submission1['id'])

sub.head(5)
col_names = Test_submission1.columns.values

sub.columns = col_names

sub.head(5)
sub.to_csv('submission1.csv', index = False)