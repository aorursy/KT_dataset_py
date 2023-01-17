import numpy as np

import pandas as pd

import os

import json

import cv2

import keras

import matplotlib.pyplot as plt
from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Lambda, Activation, Flatten, Input

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam, RMSprop, SGD

from keras.utils import np_utils
!wget https://storage.googleapis.com/kaggle-forum-message-attachments/147157/5458/bet_labels.json

!wget https://storage.googleapis.com/kaggle-forum-message-attachments/147157/5459/shark_labels.json

!wget https://storage.googleapis.com/kaggle-forum-message-attachments/147157/5460/dol_labels.json

!wget https://storage.googleapis.com/kaggle-forum-message-attachments/147157/5461/yft_labels.json

!wget https://storage.googleapis.com/kaggle-forum-message-attachments/147157/5462/alb_labels.json

!wget https://storage.googleapis.com/kaggle-forum-message-attachments/147157/5463/lag_labels.json

!wget https://storage.googleapis.com/kaggle-forum-message-attachments/147332/5471/other_labels.json

!wget https://storage.googleapis.com/kaggle-forum-message-attachments/158691/5864/NoF_labels.json
print(os.listdir("../input"))
print(os.listdir("."))
!unzip -q ../input/train.zip
trn_images = []

trn_labels = []



for category in os.listdir("train/"):

    if os.path.isdir("train/%s" % (category)):

        for img in os.listdir("train/%s" % (category)):

            if img.endswith(".jpg"):

                trn_images.append("train/%s/%s" % (category, img))

                trn_labels.append("%s" % (category))
trn_images[:5]
trn_labels[:5]
dataset = pd.DataFrame({"images": trn_images, "labels": trn_labels})

dataset.head(5)
grouped = dataset.groupby("labels")

grouped.count()
filenames = [img[0] for img in grouped.first().values]

labels = [filename.split("/")[1] for filename in filenames]

print(filenames)

print(labels)
x = []

y = []

for filename, label in zip(filenames, labels):

    img = cv2.imread(filename)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))

    x.append(img)

    y.append(label)
y
# Print images of each class

fig, axs = plt.subplots(2, 4, figsize=(25, 15))

for ax, img, label in zip(axs.flatten(), x, y):

    ax.set_title(label, fontsize=15)

    ax.imshow(img)

    ax.grid(True)



plt.show()
label2idx = {k:i for i, k in enumerate(labels)}

idx2label = {i:k for i, k in enumerate(labels)}



print(label2idx)

print(idx2label)
dataset.count()
X_train = np.zeros((3777, 224, 224, 3))

y_train = []

trn_size = np.zeros((3777, 2))

trn_filename = np.empty(3777, dtype=np.object)
print(X_train.shape, len(y_train), trn_size.shape, trn_filename.shape)
counter = 0

for rec in dataset.values:

    img = cv2.imread(rec[0])

    height, width, channel = img.shape

    img = cv2.resize(img, (224, 224))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    X_train[counter] = img

    y_train.append(label2idx.get(rec[1]))

    trn_size[counter][0] = width

    trn_size[counter][1] = height

    trn_filename[counter] = rec[0].split('/')[2]



    counter += 1



y_train = keras.utils.to_categorical(y_train, 8)

permutation = np.random.permutation(X_train.shape[0])

X_train = X_train[permutation]

y_train = y_train[permutation]

trn_size = trn_size[permutation]

trn_filename = trn_filename[permutation]
y_train[100:102]
print(X_train.shape)

print(y_train.shape)

print(trn_size[0:5])

print(trn_filename[0:5])
anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']
bb_json = {}



for c in anno_classes:

    j = json.load(open('{}_labels.json'.format(c), 'r'))

    

    for l in j:

        if 'annotations' in l.keys() and len(l['annotations']) > 0:

            bb_json[l['filename']] = l['annotations'][-1]
bb_json['img_07763.jpg']
empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}
for f in trn_filename:

    if not f in bb_json.keys(): bb_json[f] = empty_bbox
bb_json['img_07763.jpg']
def convert_bb(img, width, height):

    bb = []

    conv_x = (224. / width)

    conv_y = (224. / height)

    bb.append(bb_json[img]['height'] * conv_y)

    bb.append(bb_json[img]['width'] * conv_x)

    bb.append(max(bb_json[img]['x'] * conv_x, 0))

    bb.append(max(bb_json[img]['y'] * conv_y, 0))

    return bb
len(bb_json.keys())
trn_resize_dim = []

trn_bbox = []
for i in range(len(trn_filename)):

    trn_bbox.append(convert_bb(trn_filename[i], trn_size[i][0], trn_size[i][1]))
trn_bbox = np.asarray(trn_bbox)
print(trn_bbox[100])
print(trn_filename[100])
bb_json.get('img_03693.jpg')
input_img = Input(shape=(224, 224, 3))



x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(input_img)

x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(x)

x = MaxPooling2D((2, 2))(x)



x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)

x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)

x = MaxPooling2D((2, 2))(x)



x = Flatten()(x)

x = Dense(512, activation = 'relu')(x)

x = Dense(512, activation = 'relu')(x)



x_bb = Dense(4, name='bb')(x)

x_class = Dense(8, activation='softmax', name='class')(x)



model = Model([input_img], [x_bb, x_class])

model.compile(Adam(lr=0.001), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'])
model.summary()
model.fit(X_train, [trn_bbox, y_train], batch_size=64, epochs=10, validation_split=0.1)
test = cv2.imread('train/ALB/img_00003.jpg')

test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

test = cv2.resize(test, (224, 224))

test = test.reshape(1, 224, 224, 3)



result = model.predict(test)
result
result[0]
result[1], np.argmax(result[1])