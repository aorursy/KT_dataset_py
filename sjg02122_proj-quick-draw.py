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
df = pd.read_csv(dirname+'/'+'cat.csv')

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
bar = '□□□□□□□□□□'

sw = 1

def percent_bar(array,count,st_time):   #퍼센트를 표시해주는 함수

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

            bar = bar.replace('□','■',1)

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

    del Y     #RAM메모리 절약을 위해 사용하지 않는 변수 삭제

    

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

del tmpy     #RAM메모리 절약을 위해 사용하지 않는 변수 삭제



print(X_train.shape,X_val.shape,Y_train.shape,Y_val.shape)
K = class_num

 

 

input_tensor = Input(shape=(64, 64, 1), dtype='float32', name='input')

 

 

def conv1_layer(x):    

    x = ZeroPadding2D(padding=(3, 3))(x)

    x = Conv2D(64, (7, 7), strides=(2, 2))(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1,1))(x)

 

    return x   

 

    

 

def conv2_layer(x):         

    x = MaxPooling2D((3, 3), 2)(x)     

 

    shortcut = x

 

    for i in range(3):

        if (i == 0):

            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

            

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

 

            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)

            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)            

            x = BatchNormalization()(x)

            shortcut = BatchNormalization()(shortcut)

 

            x = Add()([x, shortcut])

            x = Activation('relu')(x)

            

            shortcut = x

 

        else:

            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

            

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

 

            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)

            x = BatchNormalization()(x)            

 

            x = Add()([x, shortcut])   

            x = Activation('relu')(x)  

 

            shortcut = x        

    

    return x

 

 

 

def conv3_layer(x):        

    shortcut = x    

    

    for i in range(4):     

        if(i == 0):            

            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)        

            

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)  

 

            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)

            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)

            x = BatchNormalization()(x)

            shortcut = BatchNormalization()(shortcut)            

 

            x = Add()([x, shortcut])    

            x = Activation('relu')(x)    

 

            shortcut = x              

        

        else:

            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

            

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

 

            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)

            x = BatchNormalization()(x)            

 

            x = Add()([x, shortcut])     

            x = Activation('relu')(x)

 

            shortcut = x      

            

    return x

 

 

 

def conv4_layer(x):

    shortcut = x        

  

    for i in range(6):     

        if(i == 0):            

            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)        

            

            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)  

 

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)

            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)

            x = BatchNormalization()(x)

            shortcut = BatchNormalization()(shortcut)

 

            x = Add()([x, shortcut]) 

            x = Activation('relu')(x)

 

            shortcut = x               

        

        else:

            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

            

            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

 

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)

            x = BatchNormalization()(x)            

 

            x = Add()([x, shortcut])    

            x = Activation('relu')(x)

 

            shortcut = x      

 

    return x

 

 

 

def conv5_layer(x):

    shortcut = x    

  

    for i in range(3):     

        if(i == 0):            

            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)        

            

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)  

 

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)

            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)

            x = BatchNormalization()(x)

            shortcut = BatchNormalization()(shortcut)            

 

            x = Add()([x, shortcut])  

            x = Activation('relu')(x)      

 

            shortcut = x               

        

        else:

            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

            

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

 

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)

            x = BatchNormalization()(x)           

            

            x = Add()([x, shortcut]) 

            x = Activation('relu')(x)       

 

            shortcut = x                  

 

    return x

 

 

 

x = conv1_layer(input_tensor)

x = conv2_layer(x)

x = conv3_layer(x)

x = conv4_layer(x)

x = conv5_layer(x)

 

x = GlobalAveragePooling2D()(x)

output_tensor = Dense(K, activation='softmax')(x)

 

resnet50 = Model(input_tensor, output_tensor)

resnet50.summary()
def top_3_accuracy(x,y): 

    t3 = top_k_categorical_accuracy(x,y, 3)

    return t3



learning_rate = 0.0001

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 

                                   verbose=0, mode='auto', min_delta=0.005, cooldown=5, min_lr=learning_rate)

earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=3) 

callbacks = [reduceLROnPlat, earlystop]



resnet50.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy', top_3_accuracy])



history = resnet50.fit(x=X_train, y=Y_train,

          batch_size = 128,

          epochs = 3,

          validation_data = (X_val, Y_val),

          callbacks = callbacks,

          verbose = 0)
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
imgs = x_test

pred = resnet50.predict(imgs, verbose=1)

top_3 = np.argsort(-pred)[:, 0:3]



#print(pred)

print(top_3)
top_3_pred = ['%s %s %s' % (class_label[k[0]], class_label[k[1]], class_label[k[2]]) for k in top_3]

print(top_3_pred[0:5])
preds_df = pd.read_csv('/kaggle/input/quickdraw-doodle-recognition/sample_submission.csv', index_col=['key_id'])

preds_df['word'] = top_3_pred

preds_df.to_csv('subcnn_small.csv')

preds_df.head()