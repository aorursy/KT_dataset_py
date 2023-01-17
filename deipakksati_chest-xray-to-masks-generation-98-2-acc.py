# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import glob

import cv2

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.layers import Dropout, Flatten, Input, Dense

from matplotlib import pyplot as plt

import numpy as np

from keras.models import *

from keras.layers import *

from keras.optimizers import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras import backend as keras

from sklearn.metrics import accuracy_score
image_path = os.path.join("/kaggle/input/chest-xray-masks-and-labels/Lung Segmentation/CXR_png")

mask_path = os.path.join("/kaggle/input/chest-xray-masks-and-labels/data/Lung Segmentation/masks")

images = os.listdir(image_path)

mask = os.listdir(mask_path)

print(len(images), len(mask))
mask2 = [fName.split(".png")[0] for fName in mask]

image_file_name = [fName.split("_mask")[0] for fName in mask2]

print(len(image_file_name), len(mask))
check = [i for i in mask if "mask" in i]

print(len(check))
dsize = (256, 256)

def image_resize(img):

    return cv2.resize(img,dsize )
#print(mask)

x = np.array([np.array(np.stack(( image_resize(cv2.imread(os.path.join(image_path,filename.split("_mask")[0]+".png"),  0)),), axis=-1)) for filename in image_file_name])

y= np.array([np.array(np.stack(( image_resize(cv2.imread(os.path.join(mask_path,filename),  0)),), axis=-1)) for filename in mask])

print(x.shape, y.shape)
def show_random_examples(x, y, p):

    indices = np.random.choice(range(x.shape[0]), 10, replace=False)

    x = x[indices]

    y = y[indices]

    p = p[indices]

    dsize2 =(400,400)

    plt.figure(figsize=(10, 5))

    for i in range(10):

        plt.subplot(2, 5, i + 1)

        xi = cv2.resize(x[i], dsize2)

        plt.imshow(xi)

        plt.xticks([])

        plt.yticks([])

#        col = 'green' if np.argmax(y[i]) == np.argmax(p[i]) else 'red'

#        plt.xlabel(class_names[np.argmax(p[i])], color=col)

    plt.show()    

    for i in range(10):

        plt.subplot(2, 5, i + 1)

        xi = cv2.resize(y[i], dsize2)

        plt.imshow(xi)

        plt.xticks([])

        plt.yticks([])

#        col = 'green' if np.argmax(y[i]) == np.argmax(p[i]) else 'red'

#        plt.xlabel(class_names[np.argmax(p[i])], color=col)

    plt.show()    

    for i in range(10):

        plt.subplot(2, 5, i + 1)

        xi = cv2.resize(p[i], dsize2)

        plt.imshow(xi)

        plt.xticks([])

        plt.yticks([])

#        col = 'green' if np.argmax(y[i]) == np.argmax(p[i]) else 'red'

#        plt.xlabel(class_names[np.argmax(p[i])], color=col)

    plt.show()   
from sklearn.model_selection import train_test_split

X_train, X_test, y_train_m, y_test_m = train_test_split(x, y, test_size=0.2, random_state=123)

x2 = np.flip(X_train , axis = 2)

y2 = np.flip(y_train_m , axis = 2)

X_train= np.append(X_train, x2,axis=0)

y_train_m = np.append(y_train_m, y2,axis=0)

print(X_train.shape,y_train_m.shape )

show_random_examples(X_train, y_train_m, y_train_m)
def unet(pretrained_weights = None,input_size = (256,256,1)):

    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)



    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    drop5 = Dropout(0.5)(conv5)



    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))

    merge6 = concatenate([drop4,up6], axis = 3)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)



    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))

    merge7 = concatenate([conv3,up7], axis = 3)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)



    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

    merge8 = concatenate([conv2,up8], axis = 3)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)



    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

    merge9 = concatenate([conv1,up9], axis = 3)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)



    model = Model(inputs, conv10)



    model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])

    

    #model.summary()



   # if(pretrained_weights):

    #    model.load_weights(pretrained_weights)



    return model
model_1 = unet()

model_1.summary()
y_test_m2 = (y_test_m/255.> .5).astype(int)
y_train_m2 = (y_train_m/255.> .5).astype(int)

model_checkpoint = ModelCheckpoint('/kaggle/working/unet_membrane_a1.hdf5', monitor='loss',verbose=2, save_best_only=True)



h = model_1.fit(

    X_train/255., y_train_m2,   

    callbacks=[model_checkpoint],

    validation_data=(X_test/255., y_test_m2),

    epochs=15, batch_size=20,

    #validation_split=0.2,

    #shuffle=True,

)
plt.plot(h.history['accuracy'])

plt.plot(h.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(h.history['loss'])

plt.plot(h.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()





losses = h.history['loss']

accs = h.history['accuracy']

val_losses = h.history['val_loss']

val_accs = h.history['val_accuracy']

epochs = len(losses)



preds = model_1.predict(X_test/255.)*255



show_random_examples(X_test, y_test_m, preds)
from keras.callbacks import ReduceLROnPlateau

learning_rate_decay = ReduceLROnPlateau(monitor='loss', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
h2 = model_1.fit(

    X_train/255., y_train_m2,

    validation_data=(X_test/255., y_test_m2),

    epochs=50, batch_size=16,

    shuffle=True,

    verbose=2,

    callbacks=[learning_rate_decay]

)

plt.plot(h2.history['accuracy'])

plt.plot(h2.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(h2.history['loss'])

plt.plot(h2.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()





losses = h2.history['loss']

accs = h2.history['accuracy']

val_losses = h2.history['val_loss']

val_accs = h2.history['val_accuracy']

epochs = len(losses)



preds2 = model_1.predict(X_test/255.)*255



show_random_examples(X_test, y_test_m, preds2)
test_path = os.path.join("/kaggle/input/chest-xray-masks-and-labels/data/Lung Segmentation/test")

test = os.listdir(test_path)

#print(test)

y= np.array([np.array(np.stack(( image_resize(cv2.imread(os.path.join(test_path,filename),  0)),), axis=-1)) for filename in test])



preds3 = model_1.predict(y/255.)*255

show_random_examples(y, y, preds3)