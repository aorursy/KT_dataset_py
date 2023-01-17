# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/guess-the-correlation-1"))
path_img = '../input/guess-the-correlation-1/train_imgs/train_imgs/'
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/guess-the-correlation-1/train_responses.csv')
print(df.shape)
df.head(5)
# Read image in Grayscale
def read_image(img):
    return io.imread(path_img+img+'.png', as_grey=True)
img_sample_10 = df.sample(9)

def plt_df_images(images):
    fig, m_ax = plt.subplots(3, 3, figsize = (15, 15))
    m_ax = m_ax.flatten()
    i = 0
    for img in images:
        m_ax[i].imshow(img, cmap = 'gray')
        #m_ax[i].axis('off')
        i = i + 1

images = [read_image(name) for name in img_sample_10.id]
plt_df_images(images)
print(images[0].shape)
gc.collect()
def crop_img(img):
    imp_crop = img[2:-21, 21:-2]
    return imp_crop

images = [crop_img(read_image(name)) for name in img_sample_10.id]
plt_df_images(images)
print(images[0].shape)
def img2data(img):
    img      = read_image(img)
    img_crop = crop_img(img)
    #gc.collect()
    data     = np.array(img_crop, dtype = 'bool')
    return data

img = img2data(df['id'][0])
img
plt.imshow(img, cmap = 'gray')
from sklearn.model_selection import train_test_split
# Read image to Matriz Transform (X)
X = np.array([img2data(name) for name in df['id']])
X.shape
# Read Correlation (y)
y = df['corr']
y[0:3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
(len(X_train), len(X_test), len(y_train), len(y_test))
X_train = X_train.reshape(X_train.shape[0], 127, 127, 1)
X_test = X_test.reshape(X_test.shape[0], 127, 127, 1)
X_train[0].shape
gc.collect()
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling2D, Flatten, AveragePooling2D
from keras.layers import LSTM, Conv2D,Activation
## Base model with Minimal CNN
def base_model():
  model = Sequential()

  model.add(Conv2D(32, (3, 3), activation='relu', 
                   input_shape=X_train[0].shape))
  model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Conv2D(64, (5, 5), activation='relu'))
  model.add(AveragePooling2D(pool_size=(4, 4)))

  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(1,   activation='linear'))

  model.compile(loss='mean_squared_error', optimizer='adam')
  return model

# Model
model = base_model()
model.summary()
batch_size = 128
nb_epoch   = 2

hist = model.fit(X_train, y_train, 
                    batch_size=batch_size, 
                    epochs=nb_epoch, verbose=1, 
                    validation_data=(X_test, y_test) )
model.save_weights('model.h5', overwrite=True)
plt.plot(hist.history)

