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
train_path = '../input/train/'

test_path = '../input/test/'

train_labels_path = '../input/train_labels.csv'

sample_path = '../input/sample_submission.csv'
td = pd.read_csv(train_labels_path,dtype=str)

testd = pd.read_csv(sample_path,dtype=str)
# d = td[td['id']=='f46f19fc90347d350431da5bfcf955d9c1418b43']['label']

# # td.label.value_counts().sum()

# print(np.int(d))

testd.head()
from PIL import Image

import matplotlib.pyplot as plt

import os

from os import path

from random import shuffle

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import math

from keras.utils import plot_model

from keras.models import Model, Sequential

from keras.layers import Input

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Activation

from keras.layers import Dropout

from keras.layers import Maximum

from keras.layers import ZeroPadding2D

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras import regularizers

from keras.layers import BatchNormalization

from keras.optimizers import Adam, SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.layers.advanced_activations import LeakyReLU

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
train_img_name = os.listdir(train_path)

for idx in tqdm(train_img_name):

    assert(idx.split('.')[1]=='tif')

print('***All pics are tif format***')



def append_(img):

    return img+'.tif'

def cut(im):

    return im[:-4]



# should only run once

td["id"]=td["id"].apply(append_)

testd["id"]=testd["id"].apply(append_)





def plot(img):

    plt.figure(figsize=(30,5))

    for i, j in enumerate(img):

        plt.subplot(2,8,i+1)

        plt.imshow(Image.open(test_path+j))

        

plot(testd.id[0:16])
# reference link https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

# implement of flow_from_dataframe

datagen=ImageDataGenerator(rescale=1./255, validation_split = 0.25)

train_generator=datagen.flow_from_dataframe(dataframe= td,

                                            directory= train_path,

                                            x_col="id", y_col="label",

                                            subset="training",

                                            seed=42,

                                            shuffle=True,

                                            class_mode="binary",

                                            target_size=(32,32),

                                            batch_size=32)

valid_generator=datagen.flow_from_dataframe(dataframe= td,

                                            directory= train_path,

                                            x_col="id", y_col="label",

                                            subset="validation",

                                            seed=42,

                                            shuffle=True,

                                            class_mode="binary",

                                            target_size=(32,32),

                                            batch_size=32)



testdatagen=ImageDataGenerator(rescale=1./255)

test_generator=testdatagen.flow_from_dataframe(dataframe= testd,

                                            directory= test_path,

                                            x_col="id", y_col=None,

                                            seed=42,

                                            shuffle=False,

                                            class_mode=None,

                                            target_size=(32,32),

                                            batch_size=32)



STEP_SIZE_TRAIN=math.ceil(train_generator.n/train_generator.batch_size)

STEP_SIZE_VALID=math.ceil(valid_generator.n/valid_generator.batch_size)

STEP_SIZE_TEST=math.ceil(test_generator.n/test_generator.batch_size)
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=(32,32,3)))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



model.compile(Adam(lr=0.0001),loss="binary_crossentropy", metrics=["accuracy"])



model.summary()

model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=40)
# You need to reset the test_generator before whenever you call the predict_generator. 

# This is important, if you forget to reset the test_generator you will get outputs in a weird order.

test_generator.reset()



pred=model.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)

pred[pred>=0.5]=1

pred[pred<0.5]=0
results=pd.DataFrame({"id":pd.read_csv(sample_path,dtype=str)['id'].apply(cut),"label":np.squeeze(pred)})

results.to_csv("results.csv",index=False)
results.head()
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "submit.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# # create a random sample dataframe

df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# create a link to download the dataframe

create_download_link(results)



# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 