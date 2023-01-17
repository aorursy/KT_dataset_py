# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time, gc, math

import cv2



from tensorflow import keras

start = time.process_time()
# PATH = 'C:\\Users\\Gaurav\\Documents\\Bengali.Ai\\bengaliai-cv19\\'

PATH = '/kaggle/input/bengaliai-cv19/'

PATH1 = '/kaggle/input/bengali-png-model/'

# train_df_ = pd.read_csv(PATH+'train.csv')

# test_df_ = pd.read_csv(PATH+'test.csv')

# class_map_df = pd.read_csv(PATH+'class_map.csv')

# sample_sub_df = pd.read_csv(PATH+'sample_submission.csv')
HEIGHT = 137

WIDTH = 236



BATCH_SIZE = 128



IMG_SIZE=64

N_CHANNELS=3
# def resize(df, size=IMG_SIZE, need_progress_bar=True):

#     resized = {}

#     X = df.values.reshape(df.shape[0], HEIGHT, WIDTH)

#     for i in range(df.shape[0]):

#         img = X[i]

# #         img = (img*(255.0/img.max())).astype(np.uint8) 

# #         img = crop_resize(img)

# #         stacked_img = np.stack((img,)*3, axis=-1)

# #         resized[df.index[i]] = stacked_img.reshape(-1)

#         img = cv2.resize(img,(size,size),interpolation=cv2.INTER_AREA)

#         resized[df.index[i]] = img.reshape(-1)

#     resized = pd.DataFrame(resized).T

#     return resized
def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size=IMG_SIZE, pad=16):

    #crop a box around pixels large than the threshold 

    #some images contain line at the sides

    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    #cropping may cut too much, so we need to add it back

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    #remove lo intensity pixels as noise

    img[img < 28] = 0

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + pad

    #make sure that the aspect ratio is kept in rescaling

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return cv2.resize(img,(size,size))
from keras.models import load_model

# load model

model = load_model(PATH1+'08_ResNet101_0.hdf5')

# summarize model.

model.summary()
components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in range(4):

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    index = df_test_img['image_id']

    index_count = 0

    df_test_img = 255 - df_test_img.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

    

    if df_test_img.shape[0] < 10000:

        split_len = 1

    else:

        split_len = 5

    split_data = {}

    split = math.ceil(df_test_img.shape[0]/split_len)

    while split > 15000:

        split_len += 1

        split = math.ceil(df_test_img.shape[0]/split_len)

        

    for j in range(split_len):

        split_data[j] = df_test_img[0:split]

        df_test_img = np.delete(df_test_img, range(split_data[j].shape[0]), 0)

    

    del df_test_img

    gc.collect()

    

    for k in range(split_len):

        preds_dict = {

            'grapheme_root': [],

            'vowel_diacritic': [],

            'consonant_diacritic': []

        }

        resized={}

        

        for l in range(split_data[k].shape[0]):

            img = split_data[k][l]

            img = (img*(255.0/img.max())).astype(np.uint8) 

            img = crop_resize(img)

            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            resized[l] = img.reshape(-1)   

        

        resized = pd.DataFrame(resized).T

        X_test = resized.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)



        preds = model.predict(X_test/255)



        for m, p in enumerate(preds_dict):

            preds_dict[p] = np.argmax(preds[m], axis=1)

        

        for n,id in enumerate(index[index_count:(index_count+split_data[k].shape[0])].values):

            for i,comp in enumerate(components):

                id_sample=id+'_'+comp

                row_id.append(id_sample)

                target.append(preds_dict[comp][n])

        

        

        index_count += split_data[k].shape[0]

        del split_data[k]

        del resized

        del preds_dict

        del preds

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
print(time.process_time() - start)