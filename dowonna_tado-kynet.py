from PIL import Image , ImageDraw

from sklearn.preprocessing import *

import time

import ast

import os

import json

import tensorflow as tf

import matplotlib.pyplot as plt

import glob

import re

import cv2

import numpy as np

import tensorflow as tf

import pandas as pd

from sklearn.model_selection import KFold

from tqdm import tqdm

from keras import layers

from keras import models 

from keras import regularizers

from keras.layers import Activation

from keras.layers import BatchNormalization 

from keras.layers import DepthwiseConv2D

from keras.layers import MaxPooling2D

import keras





from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

from keras.models import Sequential

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint







import os

fol = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        fol.append(filename)

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def make_img(img_arr) :

    image = Image.new("P", (256,256), color=255)

    image_draw = ImageDraw.Draw(image)

    for stroke in img_arr:

        for i in range(len(stroke[0])-1):

            image_draw.line([stroke[0][i], 

                             stroke[1][i],

                             stroke[0][i+1], 

                             stroke[1][i+1]],

                            fill=0, width=5)

    return image

# img = make_img(img_arr[3])

# plt.imshow(img)
def preprocessing(filenames) :

    img_batch = 2000

    X= []

    Y= []

    class_label = []

    st_time = time.time()

    class_num = 340

    Y_num = 0

    for fname in tqdm(filenames[0:class_num]) :

        #percent_bar(filenames[0:class_num],Y_num+1,st_time)

        df = pd.read_csv(os.path.join(dirname,fname))

        df['word'] = df['word'].replace(' ','_',regex = True)

        class_label.append(df['word'][0])

        keys = df.iloc[:img_batch].index

        #print(len(keys))

        

        for i in range(len(df.loc[keys,'drawing'].values)) :

            if df.loc[keys,'recognized'].values[i] == True :

                drawing = ast.literal_eval(df.loc[keys,'drawing'].values[i])

                img = make_img(drawing)

                img = np.array(img.resize((64,64)))

                img = img.reshape(64,64,1)

                X.append(img)

                Y.append(Y_num)

        Y_num += 1

        

    tmpx = np.array(X)



    Y = np.array([[i] for i in Y])

    enc = OneHotEncoder(categories='auto')

    enc.fit(Y)

    tmpy = enc.transform(Y).toarray()

    

#     del X

#     del Y     #RAM메모리 절약을 위해 사용하지 않는 변수 삭제

    

    return tmpx , tmpy , class_label , class_num



tmpx , tmpy , class_label , class_num = preprocessing(filenames)

print('\n',tmpx.shape, tmpy.shape, '\n5th class : ',class_label[0:5])

#df.head()

#print(drawing[0])

#img = make_img(drawing[1])

#plt.imshow(img)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(tmpx,tmpy, test_size = 0.1,random_state = 3)

    #RAM메모리 절약을 위해 사용하지 않는 변수 삭제



print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

del tmpx

del tmpy 
# fil = 5

# model = models.Sequential()

# #model.add(layers.Conv2D(32,(fil,fil), activation='relu', input_shape=(50,50,1),padding='same',kernel_regularizer=regularizers.l2(0.1)))

# model.add(layers.Conv2D(32,(fil,fil), activation='relu', input_shape=(64,64,1),padding='same'))

# model.add(BatchNormalization(center=True, scale=True))

# model.add(Activation('relu'))  

# model.add(DepthwiseConv2D((64,64),strides=(1,1),padding='same'))

# model.add(BatchNormalization(center=True, scale=True))

# model.add(Activation('relu'))  

# #model.add(layers.Dropout(0.2))



# model.add(layers.Conv2D(64,(fil,fil), activation='relu',padding='same'))

# model.add(BatchNormalization(center=True, scale=True))

# model.add(Activation('relu'))  

# model.add(DepthwiseConv2D((32,32),strides=(2,2),padding='same'))

# model.add(BatchNormalization(center=True, scale=True))

# model.add(Activation('relu')) 



# model.add(layers.Conv2D(128,(fil,fil), activation='relu',padding='same'))

# model.add(BatchNormalization(center=True, scale=True))

# model.add(Activation('relu'))  

# model.add(DepthwiseConv2D((16,16),strides=(2,2),padding='same'))

# model.add(BatchNormalization(center=True, scale=True))

# model.add(Activation('relu')) 



# model.add(layers.Conv2D(256,(fil,fil), activation='relu',padding='same'))

# model.add(BatchNormalization(center=True, scale=True))

# model.add(Activation('relu'))  

# model.add(DepthwiseConv2D((8,8),strides=(2,2),padding='same'))

# model.add(BatchNormalization(center=True, scale=True))

# model.add(Activation('relu')) 





# model.add(layers.Conv2D(128,(fil,fil), activation='relu',padding='same'))

# model.add(BatchNormalization(center=True, scale=True))

# model.add(Activation('relu'))  

# model.add(DepthwiseConv2D((4,4),strides=(2,2),padding='same'))

# model.add(BatchNormalization(center=True, scale=True))

# model.add(Activation('relu')) 



# model.add(layers.Flatten())

# model.add(layers.Dense(4*4*128))

# model.add(BatchNormalization(center=True, scale=True))

# model.add(layers.Dense(340,activation='softmax'))





# model.summary()
fil = 5

model = models.Sequential()

#model.add(layers.Conv2D(32,(fil,fil), activation='relu', input_shape=(50,50,1),padding='same',kernel_regularizer=regularizers.l2(0.1)))

model.add(layers.Conv2D(32,(fil,fil), activation='relu', input_shape=(64,64,1),padding='same'))

model.add(BatchNormalization(center=True, scale=True))

model.add(Activation('relu'))  

model.add(MaxPooling2D((2,2),padding='same'))

model.add(layers.Dropout(0.2))



model.add(layers.Conv2D(64,(fil,fil), activation='relu',padding='same'))

model.add(BatchNormalization(center=True, scale=True))

model.add(Activation('relu'))  

model.add(MaxPooling2D((2,2),padding='same'))

model.add(layers.Dropout(0.2))



model.add(layers.Conv2D(128,(fil,fil), activation='relu',padding='same'))

model.add(BatchNormalization(center=True, scale=True))

model.add(Activation('relu'))  

model.add(MaxPooling2D((2,2),padding='same'))

model.add(layers.Dropout(0.2))



model.add(layers.Conv2D(256,(fil,fil), activation='relu',padding='same'))

model.add(BatchNormalization(center=True, scale=True))

model.add(Activation('relu'))  

model.add(MaxPooling2D((1,1),padding='same'))

model.add(layers.Dropout(0.2))



model.add(layers.Conv2D(512,(fil,fil), activation='relu',padding='same'))

model.add(BatchNormalization(center=True, scale=True))

model.add(Activation('relu'))  

model.add(MaxPooling2D((1,1),padding='same'))

model.add(layers.Dropout(0.2))



model.add(layers.Conv2D(256,(fil,fil), activation='relu',padding='same'))

model.add(BatchNormalization(center=True, scale=True))

model.add(Activation('relu'))  

model.add(MaxPooling2D((1,1),padding='same'))

model.add(layers.Dropout(0.2))



model.add(keras.layers.GlobalAveragePooling2D(data_format=None))

model.add(layers.Dense(340,activation='softmax'))

# model.add(layers.Flatten())

# model.add(layers.Dense(4*4*256))

# model.add(BatchNormalization(center=True, scale=True))

# model.add(layers.Dense(340,activation='softmax'))



model.summary()
from tensorflow.keras.metrics import top_k_categorical_accuracy

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def top_3_accuracy(x,y): 

    t3 = top_k_categorical_accuracy(x,y, 3)

    return t3



reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 

                                   verbose=1, mode='auto', cooldown=5, min_lr=0.00025)







earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=5) 

callbacks = [reduceLROnPlat,earlystop]



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy', top_3_accuracy])

hists = []

hist  = model.fit(x=x_train, y=y_train,

          batch_size = 100,

          epochs = 50,

          validation_data = (x_test, y_test),

          callbacks = callbacks,

          verbose = 1)



hists.append(hist)
hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists], sort=True)

hist_df.index = np.arange(1, len(hist_df)+1)

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))

axs[0].plot(hist_df.val_accuracy, lw=5, label='Validation Accuracy')

axs[0].plot(hist_df.accuracy, lw=5, label='Training Accuracy')

axs[0].set_title('--acc--')

axs[0].set_ylabel('Accuracy')

axs[0].set_xlabel('Epoch')

axs[0].grid()

axs[0].legend(loc=0)

axs[1].plot(hist_df.val_loss, lw=5, label='Validation MLogLoss')

axs[1].plot(hist_df.loss, lw=5, label='Training MLogLoss')

axs[1].set_title('--Loss--')

axs[1].set_ylabel('MLogLoss')

axs[1].set_xlabel('Epoch')

axs[1].grid()

axs[1].legend(loc=0)

fig.savefig('hist.png', dpi=300)

plt.show();
print(hist_df.val_loss)
print(hist_df)
print(hist_df.loss)
# del x_train

# del y_train
scores = model.evaluate(x_test, y_test)

print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
def preprocessing_test(df) :

    X= []

    keys = df.iloc[:].index

    for i in tqdm(range(len(df.loc[keys,'drawing'].values))) :

        drawing = ast.literal_eval(df.loc[keys,'drawing'].values[i])

        img = make_img(drawing)

        img = np.array(img.resize((64,64)))

        img = img.reshape(64,64,1)

        X.append(img)

    

    tmpx = np.array(X)

    return tmpx



test = pd.read_csv(os.path.join('/kaggle/input/quickdraw-doodle-recognition', 'test_simplified.csv'))

x_test = preprocessing_test(test)

print(test.shape, x_test.shape)

test.head()
imgs = x_test

pred = model.predict(imgs, verbose=1)

top_3 = np.argsort(-pred)[:, 0:3]

print("Finished !!")



#print(pred)

print(top_3)
top_3_pred = ['%s %s %s' % (class_label[k[0]], class_label[k[1]], class_label[k[2]]) for k in top_3]

print(top_3_pred[0:5])

preds_df = pd.read_csv('/kaggle/input/quickdraw-doodle-recognition/sample_submission.csv', index_col=['key_id'])

preds_df['word'] = top_3_pred

preds_df.to_csv('subcnn_small.csv')

preds_df.head()