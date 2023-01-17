import pandas as pd

import numpy as np

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm.auto import tqdm

from glob import glob

import time, gc

import cv2



from tensorflow import keras

import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.models import clone_model

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras.models import load_model

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

from matplotlib import pyplot as plt

import seaborn as sns



IMG_SIZE=86

N_CHANNELS=1



def resize(df, size=86, need_progress_bar=True):

    resized = {}

    resize_size=86

    if need_progress_bar:

        for i in tqdm(range(df.shape[0])):

            image=df.loc[df.index[i]].values.reshape(137,236)

            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



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

            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)

            resized[df.index[i]] = resized_roi.reshape(-1)

    else:

        for i in range(df.shape[0]):

            #image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

            image=df.loc[df.index[i]].values.reshape(137,236)

            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



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

            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)

            resized[df.index[i]] = resized_roi.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized



model1 = load_model('../input/bengali-alternative/densenet_169_rooty.h5')

model2 = load_model('../input/bengali-alternative/densenet_CNN (3).h5')

model3 = load_model('../input/bengali-alternative/densenet_rooty_tooty.h5')

#model3 = load_model('../input/bengali-alternative/nasnet_model.h5')

#model4 = load_model('../input/bengali-alternative/xception.h5')

#model5 = load_model('../input/bengali-alternative/inception.h5')



preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}

components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in range(4):

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img, need_progress_bar=False)/255

    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    

    preds1 = model1.predict(X_test)

    preds2 = model2.predict(X_test)

    preds3 = model3.predict(X_test)

    #preds4 = model4.predict(X_test)

    #preds5 = model5.predict(X_test)

    #preds = [np.mean([preds1[0],preds2[0],preds3[0],preds4[0]],axis=0), np.mean([preds1[1],preds2[1],preds3[1],preds4[1]],axis=0), np.mean([preds1[2],preds2[2],preds3[2],preds4[2]],axis=0)]

    preds = [np.mean([preds1[0],preds2[0],preds3[0]],axis=0), np.mean([preds1[1],preds2[1],preds3[1]],axis=0), np.mean([preds1[2],preds2[2],preds3[2]],axis=0)]

    



    for i, p in enumerate(preds_dict):

        preds_dict[p] = np.argmax(preds[i], axis=1)



    for k,id in enumerate(df_test_img.index.values):  

        for i,comp in enumerate(components):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(preds_dict[comp][k])

    del df_test_img

    del X_test

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