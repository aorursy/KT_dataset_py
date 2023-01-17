# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install git+https://github.com/rcmalli/keras-vggface.git
from collections import defaultdict

from glob import glob

from random import choice, sample



import cv2

import numpy as np

import pandas as pd

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D

from keras.models import Model

from keras.preprocessing import image

from keras.optimizers import Adam

from keras_vggface.utils import preprocess_input

from keras_vggface.vggface import VGGFace

import h5py
train_file_path = "../input/recognizing-faces-in-the-wild/train_relationships.csv"

train_folders_path = "../input/recognizing-faces-in-the-wild/train/"

val_famillies = "F09"
all_images = glob(train_folders_path + "*/*/*.jpg")

train_images = [x for x in all_images if val_famillies not in x]

val_images = [x for x in all_images if val_famillies in x]
data=pd.read_csv(train_file_path)

data.head()
plt.figure(figsize=(20,10))

for i in range(10):

    plt.subplot(2,5,i+1)

    plt.imshow(plt.imread(train_images[i]))

    
train_person_to_images_map = defaultdict(list)

ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

for x in train_images:

    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)



val_person_to_images_map = defaultdict(list)

for x in val_images:

    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)
train_person_to_images_map['F0137/MID4']
relationships = pd.read_csv(train_file_path)

relationships = list(zip(relationships.p1.values, relationships.p2.values))

relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

relationships[:5]
train = [x for x in relationships if val_famillies not in x[0]]

val = [x for x in relationships if val_famillies in x[0]]
plt.figure(figsize=(20,10))

for i in range(2):

    lis=train_person_to_images_map[train[0][i]]

    for i in range(len(lis)):

        plt.subplot(4,5,i+1)

        plt.imshow(plt.imread(lis[i]))
def read_img(path):#读取图片,并且转为网络的输入格式

    img = image.load_img(path, target_size=(197, 197))

    img = np.array(img).astype(np.float)

    return preprocess_input(img, version=2)
plt.figure(figsize=(20,10))

lis=train_person_to_images_map[train[0][0]]

for i in range(8):

    plt.subplot(2,4,i+1)

    pro_img=read_img(lis[i])

    plt.imshow(pro_img)

     
def gen(list_tuples, person_to_images_map, batch_size=16):

    ppl = list(person_to_images_map.keys())

    while True:

        batch_tuples = sample(list_tuples, batch_size // 2)

        labels = [1] * len(batch_tuples)

        while len(batch_tuples) < batch_size:

            p1 = choice(ppl)

            p2 = choice(ppl)



            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:

                batch_tuples.append((p1, p2))

                labels.append(0)



        for x in batch_tuples:

            if not len(person_to_images_map[x[0]]):

                print(x[0])



        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]

        X1 = np.array([read_img(x) for x in X1])



        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]

        X2 = np.array([read_img(x) for x in X2])

        yield [X1, X2], labels
def baseline_model():

    input_1 = Input(shape=(197, 197, 3))

    input_2 = Input(shape=(197, 197, 3))



    base_model = VGGFace(model='resnet50', include_top=False)



    for x in base_model.layers[:-3]:

        x.trainable = True



    x1 = base_model(input_1)

    x2 = base_model(input_2)



    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])

    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])



    x3 = Subtract()([x1, x2])

    x3 = Multiply()([x3, x3])



    x1_ = Multiply()([x1, x1])

    x2_ = Multiply()([x2, x2])

    x4 = Subtract()([x1_, x2_])

    x = Concatenate(axis=-1)([x4, x3])



    x = Dense(100, activation="relu")(x)

    x = Dropout(0.01)(x)

    out = Dense(1, activation="sigmoid")(x)



    model = Model([input_1, input_2], out)



    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))



    model.summary()



    return model
# file_path = "vgg_face.h5"



# checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')



# reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)



# callbacks_list = [checkpoint, reduce_on_plateau]



# model = baseline_model()

# model.fit_generator(gen(train, train_person_to_images_map, batch_size=16), use_multiprocessing=True,

#                     validation_data=gen(val, val_person_to_images_map, batch_size=16), epochs=100, verbose=1,

#                     workers = 4, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)



# test_path = "../input/recognizing-faces-in-the-wild/test/"
# def chunker(seq, size=32):#每一次预测size个

#     return (seq[pos:pos + size] for pos in range(0, len(seq), size))





# from tqdm import tqdm



# submission = pd.read_csv('../input/recognizing-faces-in-the-wild/sample_submission.csv')



# predictions = []



# for batch in tqdm(chunker(submission.img_pair.values)):

#     print(batch)

    

#     X1 = [x.split("-")[0] for x in batch]

#     X1 = np.array([read_img(test_path + x) for x in X1])



#     X2 = [x.split("-")[1] for x in batch]

#     X2 = np.array([read_img(test_path + x) for x in X2])



#     pred = model.predict([X1, X2]).ravel().tolist()

#     predictions += pred



# submission['is_related'] = predictions



# submission.to_csv("vgg_face.csv", index=False)