!pip install imageai --upgrade

!pip install efficientnet
import os

import numpy as np

import cv2

import requests

from zipfile import ZipFile

import pandas as pd

import matplotlib.pyplot as plt

from keras import models

from imageai.Detection import ObjectDetection

import efficientnet.keras as efn

from keras.preprocessing.image import ImageDataGenerator
fb = pd.read_csv('../input/facebookdata/age_gender_fb.csv')

fb = fb.set_index('_id')
data_dir = '../input/facebookdata/fbdata/fbdata'
import pickle 



bb_d = {}

for i in range(5):

  fn = '../input/facebookdata/bb'+str(i)+'.pickle'

  with open(fn,'rb') as handle:

    b = pickle.load(handle)

    bb_d.update(b)

    

import copy



temp = copy.deepcopy(bb_d)

bb_d = {}

for k,v in temp.items():

  if k in fb.index.to_numpy():

    bb_d[k] = v
fb
def plot_hist(a,title):

    pd.Series(np.array(a)).value_counts().sort_index().plot(kind='bar',title=title)
plot_hist(fb.compress_age,'Histogram of age class')
l = [len(v) for k,v in bb_d.items()]

plot_hist(l,'Histogram of number of bounding boxes')
l = [v[0][4] for k,v in bb_d.items() if len(v)>0]

plt.hist(l,bins=40)

plt.title('Histogram of confidence');
#Random visualize image òf ids

def visualize(ids,labels=None,size=None,n_imgs=16):

    plt.figure(figsize=(15,15))

    for k in range(n_imgs):

        i = np.random.randint(len(ids))

        id = ids[i]

        img = plt.imread(os.path.join(data_dir,str(id)+'.jpg'))

        if size:

            img = cv2.resize(img,(size,size))

        plt.subplot(4,4,k+1)

        plt.imshow(img)

        if labels is not None:

            plt.title(labels[i])
def get_id_with_nbb(nb=1):

  ids = np.array(list(bb_d.keys()))

  bs = list(bb_d.values())

  l = np.array([len(b) for b in bs])



  id_nb = ids[l==nb]

  return id_nb



def get_id_with_age(ids,a):

    idx = fb.loc[ids].compress_age.to_numpy()==a

    return ids[idx]



def get_age_with_ids(ids):

    return fb.loc[ids].compress_age.to_numpy()



img_size = 128

datagen = ImageDataGenerator(

    width_shift_range=0.1,

    height_shift_range=0.1,

    rotation_range=12,

    zoom_range=0.1,

    horizontal_flip=True

)



def generate_data(img_folder,name,age,batch_size=8,is_train=True):

    

    if is_train:

      indices = np.random.permutation(len(name))

    else:

      indices = np.arange(len(name))

    n_batch = int(np.ceil(len(name)/batch_size))

    j = 0

    X,y,g = [],[],[]

    for i in range(len(name)):

      

      j += 1

      filename = name[indices[i]]

      g.append(filename)

#       new_name = img_folder + str(filename)+'.jpg'

      new_name = os.path.join(img_folder, "{}.jpg".format(filename))

#       print(new_name)

      img = cv2.imread(new_name)

      img_h, img_w, _ = np.shape(img)

      for d in bb_d[filename]:

        x1, y1, x2, y2 = d[0],d[1],d[2]+1,d[3]+1

        w,h = x2-x1,y2-y1

        xw1 = max(int(x1 - 0.4 * w), 0)

        yw1 = max(int(y1 - 0.6 * h), 0)

        xw2 = min(int(x2 + 0.4 * w), img_w - 1)

        yw2 = min(int(y2 + 0.2 * h), img_h - 1)

        img = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        break



      if is_train:

        img = datagen.random_transform(img)

      X.append(img/255)

      y.append(age[indices[i]])

      if j>=batch_size:

        yield np.array(X),np.array(y),np.array(g)

        X,y,g = [],[],[]

        j = 0

    if j >0:

      yield np.array(X),np.array(y),np.array(g)

    

def generate_data2(img_folder,name,age,batch_size=8,is_train=True):

    

    if is_train:

      indices = np.random.permutation(len(name))

    else:

      indices = np.arange(len(name))

    n_batch = int(np.ceil(len(name)/batch_size))

    j = 0

    X,y,g = [],[],[]

    for i in range(len(name)):

      

      j += 1

      filename = name[indices[i]]

      g.append(filename)

#       new_name = img_folder + str(filename)+'.jpg'

      new_name = os.path.join(img_folder, "{}.jpg".format(filename))

#       print(new_name)

      img = cv2.imread(new_name)

      img_h, img_w, _ = np.shape(img)

      for d in bb_d[filename]:

        x1, y1, x2, y2 = d[0],d[1],d[2]+1,d[3]+1

        w,h = x2-x1,y2-y1

        xw1 = max(int(x1 - 0.4 * w), 0)

        yw1 = max(int(y1 - 0.6 * h), 0)

        xw2 = min(int(x2 + 0.4 * w), img_w - 1)

        yw2 = min(int(y2 + 0.2 * h), img_h - 1)

        img = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (64, 64))

        break



      if is_train:

        img = datagen.random_transform(img)

      X.append(img)

      y.append(age[indices[i]])

      if j>=batch_size:

        yield np.array(X),np.array(y),np.array(g)

        X,y,g = [],[],[]

        j = 0

    if j >0:

      yield np.array(X),np.array(y),np.array(g)
ids = get_id_with_nbb(1)

l = get_age_with_ids(ids)

visualize(ids,l,n_imgs=16)
ids = get_id_with_nbb(0)

visualize(ids,n_imgs=16)
def get_id_with_conf(low=0,high=0.4):

  ids = [(k,v[0][4]) for k,v in bb_d.items() if len(v)==1 and v[0][4]>=low and v[0][4]<high]

  return ids
r = np.array(get_id_with_conf(0,0.4))

ids,c = r[:,0].astype(int),r[:,1]

visualize(ids,c)
r = np.array(get_id_with_conf(1,10))

ids,c = r[:,0].astype(int),r[:,1]

a = get_age_with_ids(ids)

title = [str(x)+' , '+str(y) for x,y in zip(a,c)]

visualize(ids,title)
r = np.array(get_id_with_conf(1.02,10))

train_ids,train_c = r[:,0].astype(int),r[:,1]

train_a = get_age_with_ids(train_ids)

print(len(train_ids))

visualize(train_ids,train_a)
age_ids = [get_id_with_age(train_ids,a) for a in np.arange(6)]
visualize(age_ids[0])
model1 = models.load_model('../input/model-megaage/model_imp_2.h5')
import logging

import sys

import numpy as np

from keras.models import Model

from keras.layers import Input, Activation, add, Dense, Flatten, Dropout

from keras.layers.convolutional import Conv2D, AveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2

from keras import backend as K



sys.setrecursionlimit(2 ** 20)

np.random.seed(2 ** 10)





class WideResNet:

    def __init__(self, image_size, depth=16, k=8):

        self._depth = depth

        self._k = k

        self._dropout_probability = 0

        self._weight_decay = 0.0005

        self._use_bias = False

        #self._weight_init = "he_normal"



        if K.image_data_format() == "th":

            logging.debug("image_dim_ordering = 'th'")

            self._channel_axis = 1

            self._input_shape = (3, image_size, image_size)

        else:

            logging.debug("image_dim_ordering = 'tf'")

            self._channel_axis = -1

            self._input_shape = (image_size, image_size, 3)



    # Wide residual network http://arxiv.org/abs/1605.07146

    def _wide_basic(self, n_input_plane, n_output_plane, stride):

        def f(net):

            # format of conv_params:

            #               [ [kernel_size=("kernel width", "kernel height"),

            #               strides="(stride_vertical,stride_horizontal)",

            #               padding="same" or "valid"] ]

            # B(3,3): orignal <<basic>> block

            conv_params = [[3, 3, stride, "same"],

                           [3, 3, (1, 1), "same"]]



            n_bottleneck_plane = n_output_plane



            # Residual block

            for i, v in enumerate(conv_params):

                if i == 0:

                    if n_input_plane != n_output_plane:

                        net = BatchNormalization(axis=self._channel_axis)(net)

                        net = Activation("relu")(net)

                        convs = net

                    else:

                        convs = BatchNormalization(axis=self._channel_axis)(net)

                        convs = Activation("relu")(convs)



                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),

                                          strides=v[2],

                                          padding=v[3],

                                          #kernel_initializer=self._weight_init,

                                          kernel_regularizer=l2(self._weight_decay),

                                          use_bias=self._use_bias)(convs)

                else:

                    convs = BatchNormalization(axis=self._channel_axis)(convs)

                    convs = Activation("relu")(convs)

                    if self._dropout_probability > 0:

                        convs = Dropout(self._dropout_probability)(convs)

                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),

                                          strides=v[2],

                                          padding=v[3],

                                          #kernel_initializer=self._weight_init,

                                          kernel_regularizer=l2(self._weight_decay),

                                          use_bias=self._use_bias)(convs)



            # Shortcut Connection: identity function or 1x1 convolutional

            #  (depends on difference between input & output shape - this

            #   corresponds to whether we are using the first block in each

            #   group; see _layer() ).

            if n_input_plane != n_output_plane:

                shortcut = Conv2D(n_output_plane, kernel_size=(1, 1),

                                         strides=stride,

                                         padding="same",

                                         #kernel_initializer=self._weight_init,

                                         kernel_regularizer=l2(self._weight_decay),

                                         use_bias=self._use_bias)(net)

            else:

                shortcut = net



            return add([convs, shortcut])



        return f





    # "Stacking Residual Units on the same stage"

    def _layer(self, block, n_input_plane, n_output_plane, count, stride):

        def f(net):

            net = block(n_input_plane, n_output_plane, stride)(net)

            for i in range(2, int(count + 1)):

                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)

            return net



        return f



#    def create_model(self):

    def __call__(self):

        logging.debug("Creating model...")



        assert ((self._depth - 4) % 6 == 0)

        n = (self._depth - 4) / 6



        inputs = Input(shape=self._input_shape)



        n_stages = [16, 16 * self._k, 32 * self._k, 64 * self._k]



        conv1 = Conv2D(filters=n_stages[0], kernel_size=(3, 3),

                              strides=(1, 1),

                              padding="same",

                              #kernel_initializer=self._weight_init,

                              kernel_regularizer=l2(self._weight_decay),

                              use_bias=self._use_bias)(inputs)  # "One conv at the beginning (spatial size: 32x32)"



        # Add wide residual blocks

        block_fn = self._wide_basic

        conv2 = self._layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1, 1))(conv1)

        conv3 = self._layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2, 2))(conv2)

        conv4 = self._layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2, 2))(conv3)

        batch_norm = BatchNormalization(axis=self._channel_axis)(conv4)

        relu = Activation("relu")(batch_norm)



        # Classifier block

        pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(relu)

        flatten = Flatten()(pool)

        predictions_a = Dense(units=6, #kernel_initializer=self._weight_init,

                              use_bias=self._use_bias,

                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(flatten)



        model = Model(inputs=inputs, outputs=predictions_a)

        #model = Model(inputs=inputs, outputs=predictions_a)

        return model



#model1 = WideResNet(64, depth=16, k=8)()

weight_file = '../input/model-megaage/WRN_16_8.h5'

#model1.load_weights(weight_file)
preds = [{}]*6

preds
for a in range(6):    

    for X,y,ids in generate_data(data_dir,age_ids[a],[a]*len(age_ids[a]),batch_size=2048,is_train=False):

        p = model1.predict(X[0])

        preds[a].update({k:v for k,v in zip(ids,p)})
p = {}

for a in range(6):

    p.update(preds[a])
for i in range(6):

  fn = 'pred'+'.pickle'

  with open(fn,'wb') as handle:

    pickle.dump(p,handle)
print(len(preds[5]))
def visualize_with_age(a,low,high):

    ids = [k for k,v in preds[a].items() if np.argmax(v)==a and np.max(v)>low and np.max(v)<=high]

    conf = [str(np.max(v))+' '+str(k) for k,v in preds[a].items() if np.argmax(v)==a and np.max(v)>low and np.max(v)<=high]

    visualize(ids,conf)

    print(len(ids))

    return ids

    

def hist_with_age(a):

    conf = [np.max(v) for k,v in preds[a].items() if np.argmax(v)==a]

    plt.hist(conf,bins=100)

    plt.show()
t_ids = [[]]*6
a = 5

low = 0.89

high = 1

t_ids[5] = visualize_with_age(a,low,high)
hist_with_age(5)
a = 4

low = 0.58

high = 1

t_ids[4] = visualize_with_age(a,low,high)
hist_with_age(4)
hist_with_age(3)
a = 3

low = 0.58

high = 1

t_ids[3] = visualize_with_age(a,low,high)
hist_with_age(2)
a = 2

low = 0.5

high = 1

t_ids[2] = visualize_with_age(a,low,high)
a = 1

low = 0.58

high = 1

t_ids[1] = visualize_with_age(a,low,high)
a = 0

low = 0.95

high = 1

t_ids[0] = visualize_with_age(a,low,high)
hist_with_age(0)
for i in range(6):

  fn = 'train_ids_'+str(i)+'.pickle'

  with open(fn,'wb') as handle:

    pickle.dump(t_ids[i],handle)