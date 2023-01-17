# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.models import Model

from keras.layers import *



import cv2

from tqdm import tqdm_notebook as tqdm

import zipfile

import io

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



!pip install '/kaggle/input/kerasefficientnetb3/efficientnet-1.0.0-py3-none-any.whl'

import efficientnet.keras as efn





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import os





# Any results you write to the current directory are saved as output.
from keras.applications.xception import Xception

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.resnet50 import ResNet50

from keras.applications.vgg16 import VGG16

from keras.applications.inception_v3 import InceptionV3
train_data=pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

train_data=train_data.drop(['image_id','grapheme'],axis=1)



image_data=['/kaggle/input/bengaliai-cv19/train_image_data_0.parquet',

         '/kaggle/input/bengaliai-cv19/train_image_data_1.parquet',

         '/kaggle/input/bengaliai-cv19/train_image_data_2.parquet',

         '/kaggle/input/bengaliai-cv19/train_image_data_3.parquet']

display(train_data.head())
# from matplotlib import pyplot

# def show_image(idd,batch=0):

#     data=np.matrix(test_image_data_0.iloc[idd][1:].values,dtype=np.int32).reshape(137,236)

#     pyplot.imshow(data,cmap='gray')

# show_image(idd=1)
SIZE=75


import cv2

import albumentations as A



def resize(df, size=SIZE, need_progress_bar=True):

    resized = {}

    resize_size=SIZE

    if True:

        for i in tqdm(range(df.shape[0])):

            image=df.loc[df.index[i]].values.reshape(137,236)

            augBright=A.RandomBrightnessContrast(p=1.0)

            image = augBright(image=image)['image']

            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



            idx = 0 

            ls_xmin = []

            ls_ymin = []

            ls_xmax = []

            ls_ymax = []

            for cnt in contours:

                idx += 1

                x,y,w,h = cv2.boundingRect(cnt)

                ls_xmin.append(x)

                ls_ymin.append(y)

                ls_xmax.append(x + w)

                ls_ymax.append(y + h)

            xmin = min(ls_xmin)

            ymin = min(ls_ymin)

            xmax = max(ls_xmax)

            ymax = max(ls_ymax)



            roi = image[ymin:ymax,xmin:xmax]

            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)/255

            resized[df.index[i]] = resized_roi.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized


def get_model():



     

#     model = InceptionV3(input_shape=(SIZE,SIZE,3),weights="/kaggle/input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False,)

#     model = Xception(input_shape=(SIZE,SIZE,3),weights="/kaggle/input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False,)

#     model = InceptionResNetV2(input_shape=(SIZE,SIZE,3),weights="/kaggle/input/keras-pretrained-models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False,)

#     model = ResNet50(input_shape=(SIZE,SIZE,3),weights="/kaggle/input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False,)

#     model = VGG16(input_shape=(SIZE,SIZE,3),weights="/kaggle/input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False,)

    model=efn.EfficientNetB7(input_shape=(SIZE,SIZE,3),include_top=False,weights='/kaggle/input/effi-net-b7-weights/efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')







    X=GlobalAveragePooling2D()(model.output)

    x = BatchNormalization()(X)

    x = Dropout(0.3)(x)

    x = Dense(256, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.3)(x)



    

    X_vowel = Dense(11, activation='softmax',name='vowel')(x)

    X_const=Dense(7, activation='softmax',name='const')(x)

    X_root = Dense(168, activation='softmax',name='root')(x)



    model=Model(inputs=model.inputs,outputs=[X_root,X_vowel,X_const])





    return model

model=get_model()
for layer in model.layers:

    layer.trainable=True
from keras.optimizers import Adam

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

model.compile(optimizer = Adam(lr=0.01), loss={'root': 'categorical_crossentropy',

                    'vowel': 'categorical_crossentropy',

                    'const': 'categorical_crossentropy'},metrics=[categorical_accuracy])
model.summary()
from keras.callbacks import *

TRAIN=True
import gc

gc.collect()
#Load Data 1

from sklearn.model_selection import train_test_split

if not TRAIN:

    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, min_delta=1E-30,verbose=1)

    i=0

    print("Reading data for ",i)

    train_image_data=pd.read_parquet(image_data[i])

    train_image_data=train_image_data.drop('image_id',axis=1)

    b_size=train_image_data.shape[0]

    fac=10

    half_b_size=b_size//fac

    

    print("Data Read.")

        

    

    for part in range(fac):

        

        print("Resizing for ",i,part)

    



        X1=resize(train_image_data[part*half_b_size:(part+1)*half_b_size])

        print("Resized ",i,part)

    

        print("Input Shape Before : ",X1.shape)

        X1_stacked=(X1.values.reshape(X1.shape[0],SIZE,SIZE,1))

        X1_stacked=np.stack((X1_stacked,X1_stacked,X1_stacked),axis=3).reshape(X1.shape[0],SIZE,SIZE,3,)

        print("Input Shape After : ",X1_stacked.shape)

    

    

        Y1=train_data.loc[i*part*half_b_size:(i+1)*(part+1)*half_b_size-1,:]



        X1_train,X1_test,Y1_train_root,Y1_test_root,Y1_train_vowel,Y1_test_vowel,Y1_train_const,Y1_test_const=train_test_split(X1_stacked,pd.get_dummies(Y1['grapheme_root']).values, pd.get_dummies(Y1['vowel_diacritic']).values,pd.get_dummies(Y1['consonant_diacritic']).values,test_size=0.1)

    

        model.fit( X1_train,[Y1_train_root,Y1_train_vowel,Y1_train_const,],validation_data=(X1_test,[Y1_test_root,Y1_test_vowel,Y1_test_const]),batch_size=32,epochs=10,callbacks=[rlrp])

    

        del X1,Y1   

    
#ACTUAL TRAINING

from keras.callbacks.callbacks import ModelCheckpoint



if TRAIN:

    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, min_delta=1E-30,verbose=1)

    vlbk = ModelCheckpoint(filepath='/kaggle/working/weights.hdf5', verbose=1, save_best_only=True)



    for i in range(4):

        gc.collect()

        print("Reading data for ",i)

        train_image_data=pd.read_parquet(image_data[i])

        

        train_image_data=train_image_data.drop('image_id',axis=1)

        

        b_size=train_image_data.shape[0]

        fac=10

        half_b_size=b_size//fac

    

        print("Data Read.")

        

    

        for part in range(fac):

            gc.collect()

        

            print("Resizing for ",i,part)

            X1=resize(train_image_data[part*half_b_size:(part+1)*half_b_size])

            print("Resized ",i,part)

    

            X1_stacked=(X1.values.reshape(X1.shape[0],SIZE,SIZE,1))

            X1_stacked=np.stack((X1_stacked,X1_stacked,X1_stacked),axis=3).reshape(X1.shape[0],SIZE,SIZE,3,)

            print("Input Shape After : ",X1_stacked.shape)

    

    

            Y1=train_data.loc[i*b_size+(part*half_b_size):(i*b_size)+((part+1)*half_b_size)]



            X1_train,X1_test,Y1_train_root,Y1_test_root,Y1_train_vowel,Y1_test_vowel,Y1_train_const,Y1_test_const=train_test_split(X1_stacked,pd.get_dummies(Y1['grapheme_root']).values, pd.get_dummies(Y1['vowel_diacritic']).values,pd.get_dummies(Y1['consonant_diacritic']).values,test_size=0.05)

    

            model.fit( X1_train,[Y1_train_root,Y1_train_vowel,Y1_train_const,],validation_data=(X1_test,[Y1_test_root,Y1_test_vowel,Y1_test_const]),batch_size=64,epochs=10,callbacks=[rlrp])

    

            del X1,Y1,X1_train,X1_test,Y1_train_root,Y1_test_root,Y1_train_vowel,Y1_test_vowel,Y1_train_const,Y1_test_const,X1_stacked

        del train_image_data
model.save("/kaggle/working/model.h5")
import gc

preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}



components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in range(4):

    df_test_img = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/test_image_data_{i}.parquet') 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img, need_progress_bar=False)

    

    X1_stacked=(X_test.values.reshape(X_test.shape[0],SIZE,SIZE,1))

    X1_stacked=np.stack((X1_stacked,X1_stacked,X1_stacked),axis=2).reshape(X_test.shape[0],SIZE,SIZE,3,)

    

    preds = model.predict(X1_stacked)



    for i, p in enumerate(preds_dict):

        preds_dict[p] = np.argmax(preds[i], axis=1)



    for k,id in enumerate(df_test_img.index.values):  

        for i,comp in enumerate(components):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(preds_dict[comp][k])

    del df_test_img

    del X_test,X1_stacked

    gc.collect()



df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target

    },

    columns = ['row_id','target'] 

)

df_sample.to_csv('submission.csv',index=False)

df_sample.head()






