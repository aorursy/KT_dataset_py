# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import load_model

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# loading model



model = load_model('../input/bengaliaidensenet/model_densenet.h5')

IMG_SIZE=128

N_CHANNELS=1
# Resize and center crop images with size 128 x 128. Reference: https://www.kaggle.com/iafoss/image-preprocessing-128x128

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

    return cv2.resize(img,(size, size))
import cv2



HEIGHT = 137

WIDTH = 236



def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):

    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)

    for t, stroke in enumerate(raw_strokes):

        for i in range(len(stroke[0]) - 1):

            color = 255 - min(t, 10) * 13 if time_color else 255

            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),

                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if size != BASE_SIZE:

        return cv2.resize(img, (size, size))

    else:

        return img





def resize(df, size=IMG_SIZE, need_progress_bar=True):

    resized = {}

    if need_progress_bar:

        for i in tqdm(range(df.shape[0])):

            img0 = 255 - df.loc[df.index[i]].values.reshape(HEIGHT, WIDTH).astype(np.uint8)

            #normalize each image by its max val

            img = crop_resize(img0, size=size)

            resized[df.index[i]] = img.reshape(-1)

    else:

        for i in range(df.shape[0]):

            img0 = 255 - df.loc[df.index[i]].values.reshape(HEIGHT, WIDTH).astype(np.uint8)

            #normalize each image by its max val

            img = crop_resize(img0, size=size)

            resized[df.index[i]] = img.reshape(-1)

    return pd.DataFrame(resized).T
preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}
components = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in range(4):

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img, need_progress_bar=False)

    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    

#     print(X_test.shape)

    

    preds = model.predict(X_test)

#     print(preds)



    for i, p in enumerate(preds_dict):

        preds_dict[p] = np.argmax(preds[i], axis=1)



    print(preds_dict)

    

        

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