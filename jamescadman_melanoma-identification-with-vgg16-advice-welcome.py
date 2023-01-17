# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.keras import layers, models, datasets

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import VGG16

from tensorflow.keras.applications.vgg16 import decode_predictions

import matplotlib.pyplot as plt

import os

from PIL import Image

import cv2

import pydicom
#load the data

train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
print('The training data contains columns: ')

for key in train.keys():

    print(key)
plt.hist(train['target'],align='left')

plt.title('Number of benign and malignant cases in train.csv')

plt.xticks([0,1],labels=['Benign','Malignant'])

plt.show()
hist = plt.hist(train['diagnosis'],rwidth=0.5,align='left',orientation='vertical')

plt.xticks(rotation='vertical')

plt.show()
train.diagnosis.value_counts()
#plot cases as a function of age

group = train.groupby(['target','age_approx'])['benign_malignant'].count()



train[(train['target']!=1) & (train['sex']=='female')]['age_approx'].hist(bins=15,rwidth=0.5,label='female',align='right',alpha=0.7)

train[(train['target']!=1) & (train['sex']=='male')]['age_approx'].hist(bins=15,rwidth=0.5,label='male',align='mid',alpha=0.7)

plt.title('Benign cases as a function of age and gender')

plt.legend()

plt.ylabel('frequency')

plt.xlabel('age (yrs)')

plt.show()



train[(train['target']==1) & (train['sex']=='female')]['age_approx'].hist(bins=15,rwidth=0.5,label='female',align='right',alpha=0.7)

train[(train['target']==1) & (train['sex']=='male')]['age_approx'].hist(bins=15,rwidth=0.5,label='male',align='mid',alpha=0.7)

plt.title('Malignant cases as a function of age and gender')

plt.legend()

plt.ylabel('frequency')

plt.xlabel('age (yrs)')

plt.show()

train['anatom_site_general_challenge'].hist(align='mid', rwidth=0.5)

plt.xticks(rotation='vertical')

plt.ylabel('frequency')

plt.show()
train.anatom_site_general_challenge.value_counts()
#plt.imshow the first 10 training set images

i=0

for dirname, _, filenames in os.walk('/kaggle/input/siim-isic-melanoma-classification/train/'):

    while i < 10:

        image = pydicom.dcmread(os.path.join(dirname, filenames[i]))

        plt.imshow(image.pixel_array)

        plt.xticks([])

        plt.yticks([])

        plt.show()

        i+=1
benigns = train[train.target==0]['image_name'].iloc[0:3]

malignants = train[train.target==1]['image_name'].iloc[0:3]



def preprocess_image(im):

    #vary parameters here to try and highlight features in malignant cases

    im = np.asarray(im)

    im = cv2.resize(im, (224, 224))

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im = cv2.addWeighted(im, 4, cv2.GaussianBlur(im, (0,0) , 224/5), -4 ,112)

    return im



for image in benigns:

    im = Image.open('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/' + image + '.jpg')

    im = preprocess_image(im)

    plt.figure(figsize=(3,4))

    plt.imshow(im)

    plt.title('benign')

    plt.xticks([])

    plt.yticks([])



for image in malignants:

    im = Image.open('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/' + image + '.jpg')

    im = preprocess_image(im)

    plt.figure(figsize=(3,4))

    plt.imshow(im)

    plt.title('malignant')

    plt.xticks([])

    plt.yticks([])



plt.show()
#load the data

train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')



train['image_name'] += '.jpg'



data_generator = ImageDataGenerator(rescale=1./255.,validation_split=0.25)

train_generator = data_generator.flow_from_dataframe(train,

                                                     directory='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/',

                                                     x_col='image_name',

                                                     y_col='target',

                                                     target_size=(224, 224),

                                                     class_mode='raw',

                                                     subset='training',

                                                     batch_size=24,

                                                     color_mode='rgb',

                                                     fill_mode='nearest')

valid_generator = data_generator.flow_from_dataframe(train,

                                                     directory='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/',

                                                     x_col='image_name',

                                                     y_col='target',

                                                     target_size=(224, 224),

                                                     class_mode='raw',

                                                     subset='validation',

                                                     batch_size=24,

                                                     color_mode='rgb',

                                                     fill_mode='nearest')
model = models.Sequential()

model.add(VGG16(include_top=False, pooling='avg', weights='imagenet', input_shape=(224, 224, 3), classes=2))

model.add(layers.Dense(2, activation='softmax'))

model.layers[0].trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
#try and improve predictions, by including a greater fraction of malignant cases in training sample



zeros = train[train.target==0].sample(5000)

ones = train[train.target==1]

train_df = pd.concat([ones,zeros]).sample(frac=1)
data_generator = ImageDataGenerator(rescale=1./255.,validation_split=0.25)

train_generator = data_generator.flow_from_dataframe(train_df,

                                                     directory='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/',

                                                     x_col='image_name',

                                                     y_col='target',

                                                     target_size=(224, 224),

                                                     class_mode='raw',

                                                     subset='training',

                                                     batch_size=24,

                                                     color_mode='rgb',

                                                     fill_mode='nearest')

valid_generator = data_generator.flow_from_dataframe(train_df,

                                                     directory='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/',

                                                     x_col='image_name',

                                                     y_col='target',

                                                     target_size=(224, 224),

                                                     class_mode='raw',

                                                     subset='validation',

                                                     batch_size=24,

                                                     color_mode='rgb',

                                                     fill_mode='nearest')
history = model.fit_generator(train_generator,

                    steps_per_epoch=150,

                    epochs=1,

                    validation_data=valid_generator,

                    validation_steps=30)
preds = model.predict_generator(valid_generator,steps=10)
print(preds)
for i in range(10):

    print(valid_generator[i][1])