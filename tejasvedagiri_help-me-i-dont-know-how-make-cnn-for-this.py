#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Fri Jan 19 13:59:14 2018



@author: tejas

"""

import os

import numpy as np

from PIL import Image

import tqdm

import matplotlib as plt

main_dir = "data/train/stage1_train/"

list_dir = os.listdir(main_dir)

Images = []

mask = []

Image_size = 256

for content in tqdm.tqdm(list_dir):

    sub_mask = []

    t = np.zeros((Image_size,Image_size))

    Image_id = os.listdir(main_dir+content+"/images")

    mask_id = os.listdir(main_dir+content+"/masks")

    a = Image.open(main_dir+content+"/images/"+Image_id[0]).convert("L")

    a = np.array(a.resize((Image_size,Image_size),Image.ANTIALIAS))

    Images.append(a)

    for content_mask in mask_id:

        q = Image.open(main_dir+content+"/masks/"+content_mask)

        q = np.array(q.resize((Image_size,Image_size),Image.ANTIALIAS))

        sub_mask.append(q)

        

    for test in sub_mask:

        t = test+ t

    t = np.clip(t,0,1)

    mask.append(t)



np.save("Images",Images)

np.save("Masks",mask)

'''

plt.pyplot.imshow(Images[17])

plt.pyplot.imshow(mask[17])

a = np.raray(Image.open(main_dir+list_dir[0]+"/images/"+Image_id[0]).convert("L"))

a.show()



x = []

for z in mask_id:

    q = np.array(Image.open(main_dir+list_dir[0]+"/masks/"+z))

    x.append(q)

    

t = np.zeros((360,360))

for x in xz:

    t = x+t

    

t = np.clip(t, 0, 1)

'''

#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Fri Jan 19 14:49:25 2018



@author: tejas

"""

import tensorflow as tf

import tflearn 

from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.layers.core import input_data, dropout, fully_connected

from tflearn.layers.estimator import regression

import matplotlib as plt



Img_size = 256

LR = 1e-3

Model_name = "Test"

CLasses = Img_size*Img_size



import numpy as np



Images = np.load("Images.npy")

Masks = np.load("Masks.npy")

Images.resize(670,Img_size,Img_size,1)

Masks.resize(670,Img_size*Img_size)

Images_test = Images[550:670]

Images_train = Images[0:550]

Masks_test = Masks[550:670]

Masks_train = Masks[0:550]

del Images

del Masks

'''

plt.pyplot.imshow(Images[15])

plt.pyplot.imshow(Masks[15])

'''





def Train_Model():

    tf.reset_default_graph()

    #Input for CNN

    convnet = input_data(shape=[None, Img_size, Img_size,1], name='input')

    

    convnet = conv_2d(convnet, 256, 2, activation='relu')

    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 512, 2, activation='relu')

    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 1024, 3, activation='relu')

    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 2048, 3, activation='relu')

    convnet = max_pool_2d(convnet, 3)

       

    convnet = fully_connected(convnet, CLasses, activation='softmax')

    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    

    model = tflearn.DNN(convnet, tensorboard_dir='LOG')

    return model





model = Train_Model()



model.fit({'input': Images_train}, {'targets':Masks_train}, validation_set=({'input': Images_test}, {'targets': Masks_test}),n_epoch=1, snapshot_step=500, show_metric=True, run_id=Model_name,batch_size = 2)

'''  

    #Loading Previous Models If exsist

    if os.path.exists('{}.meta'.format(Model_name)):

        model.load(Model_name)

        print("Model Loaded")

        

    else:

        print("New Model")  '''