# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import Image

import random

import numpy as np

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import glob

import os

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

import math

import tensorflow as tf

from keras.layers import Dense, Input, Dropout

from keras.models import Model

print(os.listdir("../input/gtsrb_challenge/GTSRB_Challenge"))



# Any results you write to the current directory are saved as output.
labels = np.array(os.listdir("../input/gtsrb_challenge/GTSRB_Challenge/train/"))

images = glob.glob('../input/gtsrb_challenge/GTSRB_Challenge/train/**/*.ppm')



def generator(batch_size):

    x=np.zeros((batch_size,32,32,3))

    y= np.zeros((batch_size,len(labels)))

    

    while True:

        image_names = random.sample(images,batch_size)

        for i in range(batch_size):

            image_name = image_names[i]

            parts = image_name.split(os.sep)

            label = parts[-2]

            mat = cv2.imread(image_name)

            mat = cv2.resize(mat,(32,32))

            mat = mat / 255.

            x[i] = mat

            y[i] = (labels==label).astype(int)

        yield x,y




input_tensor = Input(shape=(32, 32, 3))

x = Conv2D(8,(3,3))(input_tensor)

x = Conv2D(12,(3,3))(x)

x = Conv2D(24,(3,3))(x)

x= Flatten()(x)

x = Dense(256)(x)

x = Dropout(.25)(x)

output_tensor = Dense(len(labels), activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])

# model.summary()

model.fit_generator(generator=generator(32),steps_per_epoch=512,epochs=5,verbose=1)
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='sgd',

              metrics=['accuracy'])

model.fit_generator(generator=generator(32),steps_per_epoch=512,epochs=5,verbose=1)
from tqdm import tqdm



test_image_names = glob.glob('../input/gtsrb_challenge/GTSRB_Challenge/test/*.ppm')

x = np.zeros((len(test_image_names),32,32,3))

for i in tqdm(range(len(test_image_names))):

    test_mat = cv2.imread(test_image_names[i])

    test_mat = cv2.resize(test_mat,(32,32))

    test_mat = test_mat /255.

    x[i] = test_mat

y= model.predict(x)
label_index = np.argmax(y,axis=1)

lines = ['Filename,ClassId']

for i,test_image_name in  enumerate(test_image_names):

    name = test_image_name.split(os.sep)[-1]

    line = '{},{}'.format(name,int(labels[label_index[i]]))

    lines.append(line)

    

out_file = open('output.csv','w')

for line in lines:

    out_file.write(line+'\n')
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



csv = open('output.csv').read()

b64 = base64.b64encode(csv.encode())

payload = b64.decode()

html = '<a download="output.csv" href="data:text/csv;base64,{payload}" target="_blank">Download</a>'

html = html.format(payload=payload)

HTML(html)