from __future__ import absolute_import, division, print_function

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import random

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

import math

from __future__ import absolute_import, division, print_function

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import random

import math

import tensorflow as tf

from tensorflow import keras

import pickle

import time

import os

import itertools

from keras.models import load_model

import gc

from keras.datasets import fashion_mnist

import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,Dropout,BatchNormalization,MaxPooling2D

labels = [i[1][1] for i in pd.read_csv("../input/train.csv").iterrows() if i[1][1] != 'new_whale']

label_set = set(labels)

label_converter = {n[1]:n[0] for n in enumerate(label_set)}

label_converter_back = {n[0]:n[1] for n in enumerate(label_set)}

img_train_labels = np.array([label_converter[label] for label in labels])

test = [i for i in pd.read_csv("../input/train.csv").iterrows() if i[1]['Id'] != 'new_whale']

num_labels = len(label_set)

def gen(start,end, batch_size): 

        for n in range((end-start)//batch_size):

                images = []

                labels = []

                for _ in test[start:end][n*batch_size: batch_size+batch_size*n]: 

                        img , label = _[1][0] , _[1][1]

                        label =  label_converter[label]

                        image = cv2.resize(cv2.imread("../input/train/"+img,0), (992,512)) 

                        images.append(image)

                        labels.append(label)

                yield np.expand_dims(np.array(images),axis=3) , np_utils.to_categorical(labels,num_labels)

def gen_rand(num):

    images = []

    labels = []

    for i in range(num):

        randindx = random.randint(1,len(test)-1)

        img = test[randindx][1]['Image']

        label = test[randindx][1]['Id']

        label =  label_converter[label]

        image = cv2.resize(cv2.imread("../input/train/"+img,0), (992,512))

        images.append(image)

        labels.append(label)

    yield np.expand_dims(np.array(images),axis=3) , np_utils.to_categorical(labels,num_labels)

  

def gen_rand_batch(num,batch_size):

    for i in range(num):

        yield next(gen_rand(batch_size))

test_imgs= os.listdir('../input/test')



def gen_test(start, stop):

    num = stop - start

    images = []

    img_name = []

    test_imgs= os.listdir('../input/test')[start:stop]

    for i in range(num):

        img = test_imgs[i]

        img_name.append(img)

        image =  cv2.resize(cv2.imread("../input/test/"+img,0), (992,512))

        images.append(image)

    yield np.expand_dims(np.array(images),axis=3) , img_name

  



def gen_test_batch(num,batch_size):

    for i in range(num/batch): 

        yield next(gen_test(i*batch_size,(i+1)*(batch_size)))

        

def format_predictions(pred):

        p = np.argsort(pred)[::-1][:5]

        return ' '.join([(lambda x: label_converter_back[x])(x) for x in p])
model = Sequential([ 

    Conv2D(64 , kernel_size=(5,5), strides=(5,5),input_shape=(512,992,1)),

    MaxPooling2D(pool_size=(2, 2)),

    BatchNormalization(),

    Conv2D(32 , kernel_size=(4,4), strides=(4,4)),

    MaxPooling2D(pool_size=(2, 2)),

    BatchNormalization(),

    Conv2D(16 , kernel_size=(3,3), strides=(3,3)),

    MaxPooling2D(pool_size=(2, 2)),

    BatchNormalization(),

    Flatten(),

    Dense(100,activation='relu'),

    Dense(1152,activation='softsign'),

    Dense(num_labels, activation='softmax')

])





model.compile(loss='categorical_crossentropy',optimizer='Nadam', metrics=['categorical_accuracy'])

#model.summary()



train_num = 1500

batch = 100

#test_batch = next(gen_rand(100))

start = time.time()

odel = load_model('whale_identifier.h5')

for i in range(2):

    model.save('whale_identifier.h5') 

    model.fit_generator(gen_rand_batch(train_num,batch), validation_data=None,steps_per_epoch=train_num//batch)

end = time.time()

print(end - start)

#model.fit_generator(gen_rand_batch(train_num,batch), validation_data=None,steps_per_epoch=train_num//batch)

#model.save('whale_identifier.h5') 



#del model 

model = load_model('whale_identifier.h5')

model.summary()


# tot//batch -- > 7960//40  for optimal full batch run

def model_to_kaggle(num,batch):

    #os.system('touch eggs.csv')

    g = np.empty(shape=(0, 512, 992, 1))

    files = []

    batch_predictions = lambda P :[format_predictions(p) for p in P]

    with open('submission.csv', 'w', newline='') as csvfile:

        fieldnames = ['Image','Id']

        header_writer = csv.DictWriter(csvfile, fieldnames=["Image", "Id"])

        header_writer.writeheader()

        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        for i in range(num//batch):         

            for load in gen_test(i*batch,i*batch+batch):

                g = np.empty(shape=(0, 512, 992, 1))

                g = load[0]

                preds = batch_predictions(model.predict(g))

                for z in zip(load[1] , preds):

                    writer.writerow([z[0]] + [z[1]])



        pass

start = time.time()

model_to_kaggle(7960,40)

end = time.time()

print('Ran in {} [sec]'.format(end-start))

pd.read_csv('submission.csv')
def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(pd.read_csv('submission.csv'))
def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(d)

