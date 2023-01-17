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
import os

print(os.listdir("../input"))

import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

#from keras.optimizers import Adam

from keras.callbacks import TensorBoard



import matplotlib.pyplot as plt



from keras.models import load_model

from keras.preprocessing.image import img_to_array, array_to_img

from keras.utils.vis_utils import plot_model

import cv2

import keras as k

from PIL import Image

from torchvision import datasets, transforms
train_dataset = datasets.ImageFolder(root = '../input/data/data')

train_dataset.class_to_idx
folder_benign = '../input/data/data/benign'

folder_malignant = '../input/data/data/malignant'



read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))



# Load in pictures

ims_benign = [read(os.path.join(folder_benign, filename)) for filename in os.listdir(folder_benign)]

X_benign = np.array(ims_benign, dtype='uint8')

ims_malignant = [read(os.path.join(folder_malignant, filename)) for filename in os.listdir(folder_malignant)]

X_malignant = np.array(ims_malignant, dtype='uint8')



# Create labels

y_benign = np.zeros(X_benign.shape[0])

y_malignant = np.ones(X_malignant.shape[0])



# Merge data and shuffle it

X = np.concatenate((X_benign, X_malignant), axis = 0)

y = np.concatenate((y_benign, y_malignant), axis = 0)

s = np.arange(X.shape[0])

np.random.shuffle(s)

X = X[s]

y = y[s]

y=y.reshape(-1,1)

X.shape,y.shape

X_scaled = X/255.
im_rows = 224

im_cols = 224

batch_size = 64

im_shape = (im_rows, im_cols, 3)
##model building

model = Sequential()

#convolutional layer with rectified linear unit activation

model.add(Conv2D(filters=16, kernel_size=2,padding='same'

                , activation='relu',

                 input_shape=im_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))



#32 convolution filters used each of size 3x3

#again

model.add(Conv2D(filters=32, kernel_size=2,padding='same'

                , activation='relu'))

#64 convolution filters used each of size 3x3

#choose the best features via pooling

model.add(MaxPooling2D(pool_size=(2, 2)))



#randomly turn neurons on and off to improve convergence

model.add(Conv2D(filters=64, kernel_size=2,padding='same'

                , activation='relu'))

model.add(Dropout(0.2))

#flatten since too many dimensions, we only want a classification output

model.add(Flatten())

#fully connected to get all relevant data

model.add(Dense(300, activation='relu'))

#one more dropout for convergence' sake :) 

model.add(Dropout(0.2))

#output a softmax to squash the matrix into output probabilities

model.add(Dense(1, activation='sigmoid'))
from tensorflow.keras.optimizers import RMSprop



opt = RMSprop(lr=0.0001, decay=1e-6)



# Compile model

model.compile(

    loss='binary_crossentropy',

    optimizer=opt,

    metrics=['accuracy']

)
callbacks_list = [k.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model_log=model.fit(

    x=X_scaled, y=y, batch_size=batch_size,

    epochs=15, verbose=1,validation_split=.2

     

)
# plotting the metrics

fig = plt.figure()

plt.subplot(2,1,1)

plt.plot(model_log.history['acc'])

plt.plot(model_log.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower right')


from keras.models import load_model



model.save('model1.h5')

model.save_weights('model_weight1.h5')
#new_model=load_model('model.h5')



img_array = np.array(Image.open('../input/data/data/benign/307.jpg'))

plt.imshow(img_array)

plt.show()

print(img_array)

img2 = np.reshape(img_array,[1,224,224,3])

prediction=model.predict_classes(img2)

prediction