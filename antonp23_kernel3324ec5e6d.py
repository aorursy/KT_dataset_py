# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import pandas as pd

import numpy as np

import cv2

import matplotlib.pyplot as plt

import matplotlib.image as mpimg







import gc

from tqdm import tqdm_notebook, tnrange



import tensorflow as tf

import time

from numpy import expand_dims

from tensorflow.keras.models import load_model

model = load_model("/kaggle/input/modbenai2/modcombo2.h5")
TEST = ["test_image_data_0.parquet", "test_image_data_1.parquet", 

         "test_image_data_2.parquet",

         "test_image_data_3.parquet"]

inpath = "/kaggle/input/bengaliai-cv19/"

HEIGHT = 137

WIDTH = 236

outpath = "/kaggle/working/"
testimages = ["Test_0","Test_1","Test_2","Test_3","Test_4","Test_5","Test_6","Test_7","Test_8","Test_9","Test_10","Test_11"]

resize_size=64 

def resizeimg(img):

   

    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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



    roi = img[ymin:ymax,xmin:xmax]

    resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)

    return resized_roi
i = 0

for p in TEST:

        

     print("processing ", p)

     df = pd.read_parquet(inpath+p)

     #df = pd.read_parquet(inpath+"test_image_data_0.parquet")   

     data = df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8) 

     for idx in tqdm_notebook(range(len(df)), desc ="zipping", leave=False):

        #name = df.iloc[idx,0]

        

        #normalize each image by its max val

        #img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)

        img = (data[idx].astype(np.uint8))

        #print('resizing')

        img = resizeimg(img)

        testimages[i] = img

        i+=1

        print(i)

        #gc.collect()

        #img = da.from_array(img)

        #np.append(x_tot,((img/255.0).mean()))

        #np.append(x2_tot,((img/255.0)**2).mean())

        #print('nearly there') 

        #img = cv2.imencode('.png',img)[1]

        #test = np.append(test, img)

        #cv2.imwrite(outpath+name + '.png', img)
X_test = np.array([f for f in testimages])

X_test = expand_dims(X_test, axis=3)




testdf = pd.read_csv("/kaggle/input/bengaliai-cv19/test.csv")

testdf['target'] = 0

combres = model.predict(X_test)

graphpred = np.array(combres[0])

vowelpred = np.array(combres[1])

conspred = np.array(combres[2])

resroot = graphpred.argmax(axis=-1)

resconst = vowelpred.argmax(axis=-1)

resvowel = conspred.argmax(axis=-1)
i = testdf.query('component=="grapheme_root"').index

testdf.target.iloc[i] = resroot

j = testdf.query('component=="consonant_diacritic"').index

testdf.target.iloc[j] = resconst

k = testdf.query('component=="vowel_diacritic"').index

testdf.target.iloc[k] = resvowel
row_id = testdf['row_id']

target = testdf['target']
del testdf
submission = pd.DataFrame()

submission['row_id'] = row_id

submission['target'] = target

submission.head()

submission.to_csv('submission.csv', index=False)