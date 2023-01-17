# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from skimage import io



x_path = '/kaggle/input/hair-segmentaion/x/'

y_path = '/kaggle/input/hair-segmentaion/y/'

x_paths = os.listdir(x_path)

y_paths = os.listdir(y_path)

print("Count X:",len(x_paths))

print("Count Y:",len(y_paths))
x_paths[0],y_paths[0]
images,masks = [],[]

size = min(len(x_paths),len(y_paths))

for i in range(size):

    file = x_paths[i].replace('-org.jpg','')

    if 'a' == 'a':

        img_path = file + '-org.jpg'

        mask_path = file + '-gt.pbm'

        if img_path in x_paths and mask_path in y_paths:

            images.append( io.imread(x_path + img_path,plugin='matplotlib',as_gray = True) )

            masks.append( io.imread(y_path + mask_path,plugin='matplotlib',as_gray = True))

print("Actual data size:",len(images),len(masks))
import matplotlib.pyplot as plt



def dispplay(img):

    plt.imshow(img,cmap='gray')

    plt.show()



idx = 50



img = images[idx]

dispplay(img)



img = masks[idx]

dispplay(img)
from skimage.transform import resize

np_images = np.zeros((size,224,224,1))

np_masks = np.zeros((size,224,224,1))



for i in range(size):

    img = images[i]

    msk = masks[i]

    np_images[i] = resize(img,(224,224)).reshape((224,224,1))

    np_masks[i] = resize(msk,(224,224)).reshape((224,224,1))



sample = np_images[0].reshape((224,224))

dispplay(sample)



sample = np_masks[0].reshape((224,224))

dispplay(sample)
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout

from keras.models import Model

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.optimizers import Adam

from keras import regularizers

from keras.callbacks.callbacks import ReduceLROnPlateau
inputs = Input((224, 224, 1))

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

poo5a = MaxPooling2D(pool_size=(2, 2))(conv5)

conv5a = Conv2D(1024, (3, 3), activation='relu', padding='same')(poo5a)

conv5a = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5a)
up6a = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5a), conv5], axis=3)

conv6a = Conv2D(512, (3, 3), activation='relu', padding='same')(up6a)

conv6a = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6a)

up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)

conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)

conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)

conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)

conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)

conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)

conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)

conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)

conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

conv10 = Conv2D(1, (3, 3), activation='relu', padding='same')(conv9)



model = Model(inputs=[inputs], outputs=[conv10])

model.compile(optimizer='adam', loss='mse', metrics=['acc'])

model.summary()
epochs = 50

history = model.fit(np_images,np_masks,validation_split=0.05,epochs=epochs,batch_size=64,)

model.save('model.h5')
h = history

plt.plot(h.history['acc'])

plt.plot(h.history['val_acc'])

plt.title('Model accuracy')

plt.show()



plt.plot(h.history['loss'])

plt.plot(h.history['val_loss'])

plt.title('Model Loss')

plt.show()



plt.plot(h.history['acc'])

plt.title('Model accuracy')

plt.show()



plt.plot(h.history['loss'])

plt.title('Model Loss')

plt.show()
def predict(image):

    return model.predict(np.asarray([image]) ).reshape((224,224))

idx = 9

img = np_images[idx]

real = np_masks[idx]



dispplay(img.reshape((224,224)))

dispplay(real.reshape((224,224)))



pred = predict(img)

dispplay(pred)