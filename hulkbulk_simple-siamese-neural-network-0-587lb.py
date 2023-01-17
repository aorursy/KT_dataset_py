from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,Activation, Dropout, BatchNormalization,  GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Subtract,Multiply

from keras.models import Model, Sequential

from keras.regularizers import l2

from keras import backend as K

from keras.optimizers import Adam

from skimage.io import imshow

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import random

import cv2

import os

from tqdm import tqdm
!ls ../input/
import keras_vggface

from keras.engine import  Model

from keras.layers import Input

from keras_vggface.vggface import VGGFace


input_train_dir = "../input/train/"



def train_image():

    train_image = map(lambda f : os.path.join(input_train_dir, f) , os.listdir("%s" % input_train_dir))

    train_images = []

    for f in train_image:

        for d in os.listdir(f):

            p = os.path.join(f,d)

            train_images.extend([os.path.join(p,l) for l in os.listdir(p) ] )

    

    return train_images



train_images = train_image()





def f_path(x):

    r = []

    for i in train_images:

        if x in i:

            r.append(i)

    if r != []:

        return random.choice(r)

    return x
train_df = pd.read_csv("../input/train_relationships.csv")

train_df['p1_path'] = train_df['p1'].apply(lambda x : f_path(x) )

train_df['p2_path'] = train_df['p2'].apply(lambda x : f_path(x) )

train_df['add'] = train_df['p1'] + train_df['p2']

train_df = train_df[train_df['p1']!=train_df['p1_path']]

train_df = train_df[train_df['p2']!=train_df['p2_path']]

train_df["target"] = 1
n_shuffle = 1

train_target_0 = pd.concat([ train_df.copy() for i in range(n_shuffle)],ignore_index=True)

train_target_0['target'] = 0

p1_shuffle = train_target_0[['p1','p1_path']].values

np.random.shuffle(p1_shuffle)

train_target_0[['p1','p1_path']] = p1_shuffle

p2_shuffle = train_target_0[['p2','p2_path']].values

np.random.shuffle(p2_shuffle)

train_target_0[['p2','p2_path']] = p2_shuffle

train_target_0['add'] = train_target_0['p1'] + train_target_0['p2']

data_1 = list(train_df['add'].values)

data_0 = list(train_target_0['add'].values)

data = []

for d in data_0:

    if d in data_1:

        data.append(d)

train_target_0 = train_target_0[~train_target_0['add'].isin(data)]
train_concate = pd.concat([train_df, train_target_0], ignore_index=True)
train_concate['family1'] = train_concate['p1'].apply(lambda x : x.split('/')[0])

train_concate['family2'] = train_concate['p2'].apply(lambda x : x.split('/')[0])

# Family approach

train_concate['target'] = train_concate[['family1', 'family2']].apply(lambda x : 1 if x['family1'] == x['family2'] else 0, axis=1)
print('Size of data %d' % len(train_concate))
shape = (224,224,3)
train_concate = train_concate.sample(frac=1).reset_index(drop=True)
# We have 2 inputs, 1 for each picture

left_input = Input(shape)

right_input = Input(shape)



# We will use 2 instances of 1 network for this task

convnet = Sequential([

    Conv2D(32,3, input_shape=shape),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(),

    Conv2D(16,3),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(),

    Conv2D(8,2),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(),

    Conv2D(4,2),

    BatchNormalization(),

    Activation('relu'),

    Flatten(),

    Dense(2),

    Activation('sigmoid')

])

# Connect each 'leg' of the network to each input

# Remember, they have the same weights

encoded_l = convnet(left_input)

encoded_r = convnet(right_input)



# Getting the L1 Distance between the 2 encodings

L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))



# Add the distance function to the network

L1_distance = L1_layer([encoded_l, encoded_r])



prediction = Dense(1,activation='sigmoid')(L1_distance)

siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)



optimizer = Adam(0.001, decay=2.5e-4)

#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking

siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])


def baseline_model():

    vgg_features = VGGFace(include_top=False, input_shape=shape, pooling='avg') # pooling: None, avg or max

    input_1 = Input(shape=shape)

    input_2 = Input(shape=shape)



    base_model = VGGFace(include_top=False, input_shape=shape, pooling='avg')



    for x in base_model.layers[:-3]:

        x.trainable = True



    x1 = base_model(input_1)

    x2 = base_model(input_2)



    

    

    

    L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))



    # Add the distance function to the network

    x = L1_layer([x1, x2])

    out = Dense(1, activation="sigmoid")(x)



    model = Model([input_1, input_2], out)



    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))



    model.summary()



    return model
histories = {"loss":[], 'acc' : []}

histories_vgg = {"loss":[], 'acc' : []}
from tqdm import tqdm_notebook
batch_size = 16

for _ in tqdm_notebook(range(50), desc="Epochs ..."):

    left_data = []

    right_data = []

    targets = []

    for i in tqdm_notebook(range(len(train_concate)), desc="Files process ...", leave=False):

        dt = train_concate.iloc[i].to_dict()

        left_img = cv2.imread(dt['p1_path'])

        left_img = cv2.resize(left_img, (shape[0], shape[1]))

        left_data.append(left_img)



        right_img = cv2.imread(dt['p2_path'])

        right_img = cv2.resize(right_img, (shape[0], shape[1]))

        right_data.append(right_img)

        targets.append(dt['target'])



        if  len(left_data) % batch_size ==0:

            left_data = np.squeeze(np.array(left_data))

            right_data = np.squeeze(np.array(right_data))

            targets = np.squeeze(np.array(targets))

            history = siamese_net.train_on_batch([left_data,right_data], targets)

            #history_vgg = siamese_net_vgg.train_on_batch([left_data,right_data], targets)

            left_data = []

            right_data = []

            targets = []



    left_data = np.squeeze(np.array(left_data))

    right_data = np.squeeze(np.array(right_data))

    targets = np.squeeze(np.array(targets))

    history = siamese_net.train_on_batch([left_data,right_data], targets)

    left_data = []

    right_data = []

    targets = []

    histories['loss'].append(history[0])

    histories['acc'].append(history[1])

    #histories_vgg['loss'].append(history_vgg[0])

    #histories_vgg['acc'].append(history_vgg[1])



# summarize history for accuracy

plt.plot(histories['acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.show()
# # summarize history for accuracy

# plt.plot(histories_vgg['acc'])

# plt.title('model accuracy')

# plt.ylabel('accuracy')

# plt.xlabel('epoch')

# plt.legend(['train'], loc='upper left')

# plt.show()
test_path = "../input/test/"

test_df = pd.read_csv("../input/sample_submission.csv")

test_df['p1_path'] = test_df['img_pair'].apply(lambda x : os.path.join(test_path, x.split('-')[0] ))

test_df['p2_path'] = test_df['img_pair'].apply(lambda x : os.path.join(test_path, x.split('-')[1]))
test_img_left = []

test_img_right = []



for i in range(len(test_df)):

    

    dt = test_df.iloc[i].to_dict()

    left_img = cv2.imread(dt['p1_path'])

    left_img = cv2.resize(left_img, (shape[0], shape[1]))

    test_img_left.append(left_img)

    

    right_img = cv2.imread(dt['p2_path'])

    right_img = cv2.resize(right_img, (shape[0], shape[1]))

    test_img_right.append(right_img)

    
left_data = np.squeeze(np.array(test_img_left))

right_data = np.squeeze(np.array(test_img_right))

sub = test_df[['img_pair']]

sub['is_related'] = siamese_net.predict([left_data,right_data])

sub.to_csv('sub.csv',index=False)
# sub = test_df[['img_pair']]

# sub['is_related'] = siamese_net_vgg.predict([left_data,right_data])

# sub.to_csv('sub_vgg.csv',index=False)