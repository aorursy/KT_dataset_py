# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from PIL import Image , ImageDraw

from sklearn.preprocessing import *

import time

import ast

import os

import tensorflow as tf

from keras import models, layers

from keras import Input

from keras.models import Model, load_model

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers, initializers, regularizers, metrics

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.layers import BatchNormalization, Conv2D, Activation , AveragePooling2D

from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add

from keras.models import Sequential

from keras.metrics import top_k_categorical_accuracy

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from tqdm import tqdm



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        fpath = os.path.join(dirname, filename)
print(dirname)

print(filenames[0])
df = pd.read_csv(dirname+'/'+'bird.csv')

df['word'] = df['word'].replace(' ','_',regex = True)

print(type(df['recognized'][0]))



idx= df.iloc[:5].index

print(df.loc[idx,'recognized'].values)



for i in range(len(df.loc[idx,'drawing'].values)) :

    if df.loc[idx,'recognized'].values[i] == True :

        print(i, end=' ')



idx= df.iloc[:2000].index

T_cnt = 0

F_cnt = 0

for i in range(len(df.loc[idx,'drawing'].values)) :

    if df.loc[idx,'recognized'].values[i] == True :

        T_cnt += 1

    else : F_cnt += 1



print('\nTrue Count :',T_cnt)

print('False Count :',F_cnt)

df.head()
def check_draw(img_arr) :

    k=3

    for i in range(len(img_arr[k])):

        img = plt.plot(img_arr[k][i][0],img_arr[k][i][1])

        plt.scatter(img_arr[k][i][0],img_arr[k][i][1])

    plt.xlim(0,256)

    plt.ylim(0,256)

    plt.gca().invert_yaxis()



ten_ids = df.iloc[:10].index

img_arr = [ast.literal_eval(lst) for lst in df.loc[ten_ids,'drawing'].values]  #ast.literal_eval is squence data made string to array

print(img_arr[3])

check_draw(img_arr)
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

img = make_img(img_arr[3])

img = img.resize((64,64))

plt.imshow(img)
bar = '??????????????????????????????'

sw = 1

def percent_bar(array,count,st_time):   #???????????? ??????????????? ??????

    global bar

    global sw

    length = len(array)

    percent = (count/length)*100

    spend_time = time.time()-st_time

    if count == 1 :

        print('preprocessing...')

    print('\r'+bar+'%3s'%str(int(percent))+'% '+str(count)+'/'+str(length),'%.2f'%(spend_time)+'sec',end='')

    if sw == 1 :

        if int(percent) % 10 == 0 :

            bar = bar.replace('???','???',1)

            sw = 0

    elif sw == 0 :

        if int(percent) % 10 != 0 :

            sw = 1
def preprocessing(filenames) :

    img_batch = 2000

    X= []

    Y= []

    class_label = []

    st_time = time.time()

    class_num = 340

    Y_num = 0

    for fname in filenames[0:class_num] :

        percent_bar(filenames[0:class_num],Y_num+1,st_time)

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

    

    del X

    del Y     #RAM????????? ????????? ?????? ???????????? ?????? ?????? ??????

    

    return tmpx , tmpy , class_label , class_num



tmpx , tmpy , class_label , class_num = preprocessing(filenames)

print('\n',tmpx.shape, tmpy.shape, '\n5th class : ',class_label[0:5])

#df.head()

#print(drawing[0])

#img = make_img(drawing[1])

#plt.imshow(img)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(tmpx,tmpy, test_size = 0.1,random_state = 0)

del tmpx

del tmpy     #RAM????????? ????????? ?????? ???????????? ?????? ?????? ??????



print(X_train.shape,X_val.shape,Y_train.shape,Y_val.shape)
#CNN

inputs = (64,64,1)

st_filter = 32

filter_size = (3,3) 

CNN = Sequential()

CNN.add(layers.Conv2D(st_filter,filter_size, input_shape = inputs ,padding= 'same'))

CNN.add(BatchNormalization())

CNN.add(Activation('relu'))

CNN.add(layers.MaxPooling2D((2,2),padding= 'same'))

CNN.add(layers.Dropout(0.25))

CNN.add(layers.Conv2D(st_filter*2,filter_size, input_shape = inputs,padding= 'same'))

CNN.add(BatchNormalization())

CNN.add(Activation('relu'))

CNN.add(layers.AveragePooling2D((2,2),padding= 'same'))

CNN.add(layers.Dropout(0.25))

CNN.add(layers.Conv2D(st_filter*4,filter_size, input_shape = inputs,padding= 'same'))

CNN.add(BatchNormalization())

CNN.add(Activation('relu'))

CNN.add(layers.AveragePooling2D((2,2),padding= 'same'))

CNN.add(layers.Dropout(0.25))

CNN.add(layers.Conv2D(st_filter*8,filter_size, input_shape = inputs,padding= 'same'))

CNN.add(BatchNormalization())

CNN.add(Activation('relu'))

CNN.add(layers.AveragePooling2D((2,2),padding= 'same'))

CNN.add(layers.Dropout(0.25))

CNN.add(layers.Conv2D(st_filter*16,filter_size, input_shape = inputs,padding= 'same'))

CNN.add(BatchNormalization())

CNN.add(layers.AveragePooling2D((2,2),padding= 'same'))

CNN.add(layers.Flatten())

CNN.add(layers.Dense(2*2*512,activation = 'relu'))

CNN.add(layers.Dropout(0.5))

CNN.add(layers.Dense(class_num, activation = 'softmax'))



CNN.summary()
def top_3_accuracy(x,y): 

    t3 = top_k_categorical_accuracy(x,y, 3)

    return t3



learning_rate = 0.0001

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 

                                   verbose=1, mode='auto', min_delta=0.005, cooldown=5, min_lr=learning_rate)

earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=4) 

callbacks = [reduceLROnPlat]



CNN.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy', top_3_accuracy])



history = CNN.fit(x=X_train, y=Y_train,

          batch_size = 128,

          epochs = 100,

          validation_data = (X_val, Y_val),

          callbacks = callbacks,

          verbose = 1)



#drop out no -> 0.75 0.90
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1,len(acc) + 1 )



plt.plot(epochs, acc, 'bo' , label = 'Training Accuracy')

plt.plot(epochs, val_acc, 'b' , label = 'Validation Accuracy')

plt.title('Training and Validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo' , label = 'Training Loss')

plt.plot(epochs, val_loss, 'b' , label = 'Validation Loss')

plt.title('Training and Validation Loss')

plt.legend()



plt.show()
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
plt.imshow(x_test[0].reshape(64,64))
imgs = x_test

pred = CNN.predict(imgs, verbose=1)

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