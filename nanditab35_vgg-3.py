# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import matplotlib.pyplot as plt

import gc

import scipy.io as sio

#import cv2

#import imutils

from PIL import Image

import tensorflow as tf

import tensorflow.image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("/kaggle/input"))



path = '/kaggle/input/'



# Any results you write to the current directory are saved as output.

%env JOBLIB_TEMP_FOLDER=/tmp

gc.collect()


df_train_lbl = pd.read_csv(path + 'train.csv')

df_test_lbl = pd.read_csv(path + 'test.csv')



m_tr = np.shape(df_train_lbl)[0]

m_te = np.shape(df_test_lbl)[0]



print(m_tr)



no_dr_ratio = float((np.shape(df_train_lbl.loc[df_train_lbl['diagnosis']==0])[0])/m_tr)

print(no_dr_ratio)



mild_dr_ratio = float((np.shape(df_train_lbl.loc[df_train_lbl['diagnosis']==1])[0])/m_tr)

print(mild_dr_ratio)



moderate_dr_ratio = float((np.shape(df_train_lbl.loc[df_train_lbl['diagnosis']==2])[0])/m_tr)

print(moderate_dr_ratio)



severe_dr_ratio = float((np.shape(df_train_lbl.loc[df_train_lbl['diagnosis']==3])[0])/m_tr)

print(severe_dr_ratio)



proliferative_dr_ratio = float((np.shape(df_train_lbl.loc[df_train_lbl['diagnosis']==4])[0])/m_tr)

print(proliferative_dr_ratio)
# Under-sampling of dataset

train_path = path + 'train_images/'

test_path = path + 'test_images/'



all_images = glob.glob(train_path + '*.png')

i=1



sampled_train_lbl = pd.DataFrame(columns = df_train_lbl.columns)



if 'df_train_lbl' in globals():

    for img_path in all_images[0:2]:

        base_name = os.path.basename(img_path)

        base_name = base_name.split('.')[0]



        diagnosis_lbl = df_train_lbl.loc[df_train_lbl['id_code'] == base_name]['diagnosis']

        diagnosis_lbl = diagnosis_lbl.values[0]

        plt.figure(i)

        img = plt.imread(img_path)

        plt.imshow(img)

        plt.title(str(diagnosis_lbl))

        i+=1

        

        smallest_class_ratio = 0

        if 'no_dr_ratio' in globals() and 'mild_dr_ratio' in globals() and 'moderate_dr_ratio' in globals() and 'severe_dr_ratio' in globals() and 'proliferative_dr_ratio' in globals():

            smallest_class_ratio = severe_dr_ratio

            class_size = int(m_tr * smallest_class_ratio)



            print(smallest_class_ratio)

            df_nodr = df_train_lbl[df_train_lbl.diagnosis == 0][np.random.rand(df_train_lbl[df_train_lbl.diagnosis== 0].index.size) < (smallest_class_ratio/2)]

            sampled_train_lbl = pd.concat([sampled_train_lbl,df_nodr])

            print('df_nodr size after sampling:' + str(np.shape(df_nodr)))



            df_mild_dr = df_train_lbl[df_train_lbl.diagnosis == 1][np.random.rand(df_train_lbl[df_train_lbl.diagnosis== 1].index.size) < smallest_class_ratio]

            sampled_train_lbl = pd.concat([sampled_train_lbl,df_mild_dr])

            print('df_mild_dr size after sampling:' + str(np.shape(df_mild_dr)))



            df_moderate_dr = df_train_lbl[df_train_lbl.diagnosis == 2][np.random.rand(df_train_lbl[df_train_lbl.diagnosis== 2].index.size) < smallest_class_ratio]

            sampled_train_lbl = pd.concat([sampled_train_lbl,df_moderate_dr])

            print('df_moderate_dr size after sampling:' + str(np.shape(df_moderate_dr)))



            df_severe_dr = df_train_lbl[df_train_lbl.diagnosis == 3]#[np.random.rand(df_train_lbl[df_train_lbl.diagnosis== 3].index.size) < smallest_class_ratio]

            sampled_train_lbl = pd.concat([sampled_train_lbl,df_severe_dr])

            print('df_severe_dr size after sampling:' + str(np.shape(df_severe_dr)))



            df_proliferative_dr = df_train_lbl[df_train_lbl.diagnosis == 4]#[np.random.rand(df_train_lbl[df_train_lbl.diagnosis== 4].index.size) < smallest_class_ratio]

            sampled_train_lbl = pd.concat([sampled_train_lbl,df_proliferative_dr])

            print('df_proliferative_dr size after sampling:' + str(np.shape(df_proliferative_dr)))



            print(np.shape(sampled_train_lbl))

            print(sampled_train_lbl)

            sampled_m_tr = np.shape(sampled_train_lbl)[0]



            no_dr_ratio = float((np.shape(sampled_train_lbl.loc[sampled_train_lbl['diagnosis']==0])[0])/sampled_m_tr)

            print(no_dr_ratio)



            mild_dr_ratio = float((np.shape(sampled_train_lbl.loc[sampled_train_lbl['diagnosis']==1])[0])/sampled_m_tr)

            print(mild_dr_ratio)



            moderate_dr_ratio = float((np.shape(sampled_train_lbl.loc[sampled_train_lbl['diagnosis']==2])[0])/sampled_m_tr)

            print(moderate_dr_ratio)



            severe_dr_ratio = float((np.shape(sampled_train_lbl.loc[sampled_train_lbl['diagnosis']==3])[0])/sampled_m_tr)

            print(severe_dr_ratio)



            proliferative_dr_ratio = float((np.shape(sampled_train_lbl.loc[sampled_train_lbl['diagnosis']==4])[0])/sampled_m_tr)

            print(proliferative_dr_ratio)



            print(np.shape(sampled_train_lbl))

            

            del df_nodr

            del df_mild_dr

            del df_moderate_dr

            del df_severe_dr

            del df_proliferative_dr

            

            gc.collect()



        else:

            print('variables not defined.. please run data_prep.py before running this file...')



gc.collect()

def image_resize_tf(img_path):

    filename = tf.placeholder(tf.string, name="inputFile")

    fileContent = tf.read_file(filename, name="loadFile")

    image = tf.image.decode_png(fileContent, name="decodePng")

    

    resize_nearest_neighbor = tf.image.resize_images(image, size=[224,224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    

    sess = tf.Session()

    feed_dict={filename: img_path}

    with sess.as_default():

        actual_resize_nearest_neighbor = resize_nearest_neighbor.eval(feed_dict)

        #plt.imshow(actual_resize_nearest_neighbor)

    return actual_resize_nearest_neighbor
resized_img = image_resize_tf("../input/train_images/875d2ffcbf47.png")
# shuffling the data

from random import shuffle



idx_arr = [i for i in range(0,sampled_m_tr)]

shuffle(idx_arr)

m_train_validate = int(sampled_m_tr*0.7)

m_validate = sampled_m_tr - m_train_validate

idx_train = idx_arr[:m_train_validate]

idx_validate = idx_arr[m_train_validate:]

# resizing training images

img_arr_train = np.ndarray(shape=(m_train_validate, 224, 224, 3))

lbl_train = np.ndarray(shape=(m_train_validate, 5))

#one_hot_targets = np.eye(nb_classes)[targets]

idx = 0

k = 0

for row in sampled_train_lbl.iterrows():

    if idx in idx_train:

        name = row[1]['id_code'] + '.png'

        lbl_train[k,:] = np.eye(5)[int(row[1]['diagnosis'])].T

        img = image_resize_tf(train_path + name)

        #print(img)

        img_arr_train[k,:,:,:] = img

        k += 1

    idx += 1

print(np.shape(img_arr_train))

print(np.shape(lbl_train))
# resizing validating images

img_arr_validate = np.ndarray(shape=(m_validate, 224, 224, 3))

lbl_validate = np.ndarray(shape=(m_validate, 5))

idx = 0

k = 0

for row in sampled_train_lbl.iterrows():

    if idx in idx_validate:

        name = row[1]['id_code'] + '.png'

        lbl_validate[k] = np.eye(5)[int(row[1]['diagnosis'])].T

        img = image_resize_tf(train_path + name)

        #print(img)

        img_arr_validate[k,:,:,:] = img

        k += 1

    idx += 1

print(np.shape(img_arr_validate))

print(np.shape(lbl_validate))
# vgg19 code

import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.optimizers import SGD

from keras.optimizers import Adam



input_shape = (224, 224, 3)



#Instantiate an empty model

model = Sequential([

Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),

Conv2D(64, (3, 3), activation='relu', padding='same'),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Conv2D(128, (3, 3), activation='relu', padding='same'),

Conv2D(128, (3, 3), activation='relu', padding='same',),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Conv2D(256, (3, 3), activation='relu', padding='same',),

Conv2D(256, (3, 3), activation='relu', padding='same',),

Conv2D(256, (3, 3), activation='relu', padding='same',),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Conv2D(512, (3, 3), activation='relu', padding='same',),

Conv2D(512, (3, 3), activation='relu', padding='same',),

Conv2D(512, (3, 3), activation='relu', padding='same',),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Conv2D(512, (3, 3), activation='relu', padding='same',),

Conv2D(512, (3, 3), activation='relu', padding='same',),

Conv2D(512, (3, 3), activation='relu', padding='same',),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Flatten(),

Dense(4096, activation='relu'),

Dense(4096, activation='relu'),

#Dense(1000, activation='relu'),

Dense(5, activation='softmax'),    

])



model.summary()



# Compile the model

#model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

sgd = SGD(lr=0.0001, momentum=0.9)

#adm = Adam()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='rmsprop', metrics=["accuracy"])
#img_arr_train = img_arr_train/255

#img_arr_validate = img_arr_validate/255



# contering the the image array for training

from numpy import asarray



k = 0

for k in range(0,np.size(img_arr_train,1)):

    img_train = img_arr_train[k,:,:,:]

    img_train_scaled = asarray(img_train)

    mean1, std1 = img_train_scaled.mean(), img_train_scaled.std()

    img_train_scaled = (img_train_scaled - mean1)/std1

    img_arr_train[k,:,:,:] = img_train_scaled

    

print(np.shape(img_arr_train))
# contering the the image array for validating

from numpy import asarray



k = 0

for k in range(0,np.size(img_arr_validate,1)):

    img_validate = img_arr_validate[k,:,:,:]

    img_validate_scaled = asarray(img_validate)

    mean1, std1 = img_validate_scaled.mean(), img_validate_scaled.std()

    img_validate_scaled = (img_validate_scaled - mean1)/std1

    img_arr_validate[k,:,:,:] = img_validate_scaled

    

print(np.shape(img_arr_validate))
lbls = np.array(sampled_train_lbl['diagnosis']).reshape(sampled_m_tr,1)

history = model.fit(x=img_arr_train,y=lbl_train,validation_data=(img_arr_validate, lbl_validate),batch_size=128,epochs=20,verbose=1) #batch_size=20
#img_arr_train

del img_arr_train

del img_arr_validate

del df_train_lbl

del df_test_lbl

gc.collect()
#test_images = glob.glob(test_path + '*.png')

df_sample_sub = pd.read_csv(path + 'sample_submission.csv')

m_test = np.shape(df_sample_sub)[0]

test_images = np.ndarray(shape=(m_test, 224, 224, 3))

k = 0



for row in df_sample_sub.iterrows():

    name = row[1]['id_code'] + '.png'

    img = image_resize_tf(test_path + name)

    img_test_scaled = np.asarray(img)

    mean1, std1 = img_test_scaled.mean(), img_test_scaled.std()

    img_test_scaled = (img_test_scaled - mean1)/std1

    test_images[k,:,:,:] = img_test_scaled

    k += 1
scores = model.predict_proba(test_images)

y_test = np.argmax(scores,axis=1)

print(y_test)
df_sample_sub['diagnosis'] = y_test

df_sample_sub['diagnosis'].astype('int64')
os.chdir("/kaggle/working/")

df_sample_sub.to_csv('submission.csv', index=False)