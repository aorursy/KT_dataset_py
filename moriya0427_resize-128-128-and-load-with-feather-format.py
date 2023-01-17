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

import cv2

import gc

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

import time
start_time = time.time()

data0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')

data1 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')

data2 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_2.parquet')

data3 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_3.parquet')

print("--- %s seconds ---" % (time.time() - start_time))
HEIGHT = 137

WIDTH = 236

SIZE = 128
def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size=SIZE, pad=16):

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
def Resize(df,size=SIZE):

    resized = {} 

    df = df.set_index('image_id')

    data = 255 - df.iloc[:, 0:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

    for idx in tqdm(range(df.shape[0])):

        img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)

        img = crop_resize(img, size = 128)

        resized[df.index[idx]] = img.reshape(-1)

    resized = pd.DataFrame(resized).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized
data0 = Resize(data0)

data0.to_feather('train_data_0.feather')

del data0

data1 = Resize(data1)

data1.to_feather('train_data_1.feather')

del data1

data2 = Resize(data2)

data2.to_feather('train_data_2.feather')

del data2

data3 = Resize(data3)

data3.to_feather('train_data_3.feather')

del data3
start_time = time.time()

data0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_0.parquet')

data1 = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_1.parquet')

data2 = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_2.parquet')

data3 = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_3.parquet')

print("--- %s seconds ---" % (time.time() - start_time))
data0 = Resize(data0)

data0.to_feather('test_data_0.feather')

del data0

data1 = Resize(data1)

data1.to_feather('test_data_1.feather')

del data1

data2 = Resize(data2)

data2.to_feather('test_data_2.feather')

del data2

data3 = Resize(data3)

data3.to_feather('test_data_3.feather')

del data3
start_time = time.time()

data0 = pd.read_feather('train_data_0.feather')

data1 = pd.read_feather('train_data_1.feather')

data2 = pd.read_feather('train_data_2.feather')

data3 = pd.read_feather('train_data_3.feather')

print("--- %s seconds ---" % (time.time() - start_time))
def Grapheme_plot(df):

    df_sample = df.sample(15)

    im_id, img = df_sample.iloc[:,0].values,df_sample.iloc[:,1:].values.astype(np.float)

    

    fig,ax = plt.subplots(3,5,figsize=(20,20))

    for i in range(15):

        j=i%3

        k=i//3

        ax[j,k].imshow(img[i].reshape(SIZE,SIZE), cmap='gray')

        ax[j,k].set_title(im_id[i],fontsize=20)

    plt.tight_layout()

        
Grapheme_plot(data0)

Grapheme_plot(data1)

Grapheme_plot(data2)

Grapheme_plot(data3)