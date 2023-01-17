# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir('../input'))



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

from sklearn.metrics import mean_squared_error
data_dir = '../input/'

os.listdir(data_dir)
img_dir = data_dir+"headcount/image_data/"

len(os.listdir(img_dir))
train = pd.read_csv(data_dir+'headcount/train.csv')

print(train.head())
from PIL import Image

#Resize image

#iloc[row,column]

#If you have an L mode image, that means it is a single channel image - normally interpreted as greyscale



train_images = np.array(train.iloc[:,0])

trainimagearr = []



for i in train_images:

    img = Image.open(img_dir+i).convert('L').resize((256,256))

    trainimagearr.append(np.array(img))

    
train_img = np.array(trainimagearr)

print(train_img.shape)
plt.imshow(train_img[0])
#The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1

train_img = train_img.reshape(-1,256,256,1)

print(train_img.shape)
from keras.models import Sequential

import keras

from keras.layers import Dense

from keras import layers

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D,Activation

from keras.callbacks import EarlyStopping

from keras.optimizers import RMSprop, Adam, Adadelta

import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [EarlyStopping(monitor='val_loss', patience=2),

             ModelCheckpoint(filepath='best_model.hd5', monitor='val_loss', save_best_only=True)]
from keras.applications.mobilenet_v2 import MobileNetV2

#Load the VGG model

vgg_conv = MobileNetV2(weights=None, include_top=False, input_shape=(256, 256,1))
def vgg_custom():

    model = Sequential()

    #add vgg conv model

    model.add(vgg_conv)

    

    #add new layers

    model.add(Flatten())

    model.add(Dense(1,  kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer=Adadelta())

    

    return model
mjj = vgg_custom()

mjj.summary()
reg=KerasRegressor(build_fn=vgg_custom, epochs=100, batch_size=8,verbose=1,callbacks=callbacks)

reg.fit(train_img, train.HeadCount)
#Test data process

test = pd.read_csv(data_dir+'testface/testrain/test_data.csv')

test.head()

test_images=np.array(test.iloc[:,0])
testimagearr=[]

for i in test_images:

    img=Image.open(img_dir+i).convert('L').resize((256,256))

    testimagearr.append(np.array(img))
test_img=np.array(testimagearr)
test_img=test_img.reshape(-1, 256, 256, 1)
#prediction

predict = reg.predict(test_img,verbose=1)
predict_read = pd.DataFrame({'Name':test.Name, 'HeadCount':predict})

predict_read.sample(20)
predict_read.to_csv('submission.csv')
import pickle

#Saving model

pickle.dump(reg, open('model.pkl','wb'))