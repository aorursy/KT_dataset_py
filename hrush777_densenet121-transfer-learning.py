# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
str_ = 'Hi Kagglers'

os.system('echo '+str_)
x_train = np.load('../input/reducing-image-sizes-to-32x32/X_train.npy')

x_test = np.load('../input/reducing-image-sizes-to-32x32/X_test.npy')

y_train = np.load('../input/reducing-image-sizes-to-32x32/y_train.npy')



print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255.

x_test /= 255.
y_train.shape
# from sklearn.model_selection import train_test_split

# X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, 

#                                                     test_size=0.2, 

#                                                     random_state=0)
from keras.preprocessing.image import ImageDataGenerator



# create and configure augmented image generator

datagen_train = ImageDataGenerator(

    width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)

    height_shift_range=0.1,  # randomly shift images vertically (10% of total height)

    horizontal_flip=True) # randomly flip images horizontally



# fit augmented image generator on data

datagen_train.fit(x_train)
from keras.applications import DenseNet121

from keras.layers import *

from keras.models import Sequential
conv_base = DenseNet121(weights='imagenet',include_top=False,input_shape=(32,32,3))
model = Sequential()

model.add(conv_base)

model.add(GlobalAveragePooling2D())

model.add(Dropout(0.5))

model.add(Dense(14, activation='softmax'))

model.summary()
# compile the model

model.compile(loss='categorical_crossentropy', optimizer='adam', 

                  metrics=['accuracy'])
str_ = 'Traning Started'

os.system('echo '+str_)
from keras.callbacks import ModelCheckpoint   



batch_size = 128

epochs = 25



checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_acc', 

    verbose=1, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



history = model.fit(

    x=x_train,

    y=y_train,

    batch_size=64,

    epochs=10,

    callbacks=[checkpoint],

    validation_split=0.1

)
str_ = 'Traning Ended'

os.system('echo '+str_)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['train','validation'])

plt.show()
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.legend(['train','validation'])

plt.show()
model.load_weights('model.h5')
str_ = 'Weights loaded successfully'

os.system('echo '+str_)
pred = model.predict_classes(x_test,verbose=1)
str_ = 'Prediction complete'

os.system('echo '+str_)
sam_sub = pd.read_csv('../input/iwildcam-2019-fgvc6/sample_submission.csv')

sam_sub.head()
_id = sam_sub['Id'].values

_id.shape
_id = _id.reshape(-1,1)

_id.shape
pred.shape
pred = pred.reshape(-1,1)

pred.shape
output = np.array(np.concatenate((_id, pred), 1))



output = pd.DataFrame(output,columns = ["Id","Predicted"])



output.to_csv('submission.csv',index = False)
