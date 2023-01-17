# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

os.listdir('../input/digit-recognizer')

# Any results you write to the current directory are saved as output.
PATH = '../input/digit-recognizer'



df_train = pd.read_csv(os.path.join(PATH, 'train.csv'))

train_y = df_train['label'].values

train_x = df_train.drop(['label'], axis=1).values





df_test = pd.read_csv(os.path.join(PATH, 'test.csv'))

test_x = df_test.values





print(train_x.shape)

print(train_y.shape)

print(test_x.shape)
IMG_SIZE = 32
# resize

import cv2



def resize(img_array):

    tmp = np.empty((img_array.shape[0], IMG_SIZE, IMG_SIZE))



    for i in range(len(img_array)):

        img = img_array[i].reshape(28, 28).astype('uint8')

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = img.astype('float32')/255

        tmp[i] = img

        

    return tmp



train_x_resize = resize(train_x)

test_x_resize = resize(test_x)
train_x_final = np.stack((train_x_resize,)*3, axis=-1)

test_x_final = np.stack((test_x_resize,)*3, axis=-1)

print(train_x_final.shape)

print(test_x_final.shape)
from keras.utils import to_categorical

train_y_final = to_categorical(train_y, num_classes=10)

print(train_y_final.shape)
# models 

from keras.models import Sequential

from keras.applications import VGG19

from keras.layers import Dense, Flatten



vgg19 = VGG19(weights = 'imagenet', 

              include_top = False,

              input_shape=(IMG_SIZE, IMG_SIZE, 3)

              )



model = Sequential()

model.add(vgg19)

model.add(Flatten())

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', 

              optimizer='sgd', 

              metrics=['accuracy'])



model.summary()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_x_final, train_y_final, test_size=0.2, random_state=2019)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
# callback

from keras.callbacks import ModelCheckpoint, EarlyStopping

es = EarlyStopping(monitor='val_acc', verbose=1, patience=5)

mc = ModelCheckpoint(filepath='mnist-vgg19.h5', verbose=1, monitor='val_acc')

cb = [es, mc]
history = model.fit(x_train, y_train, 

                    epochs=100, 

                    batch_size=128, 

                    validation_data=(x_test, y_test),

                    callbacks=cb)
preds = model.predict(test_x_final, batch_size=128)
preds.shape
results = np.argmax(preds, axis=-1)

results.shape
# submission

sub = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))

sub.head()

df = pd.DataFrame({'ImageId': sub['ImageId'], 'Label': results})

df.to_csv('submission.csv', index=False)

os.listdir('./')