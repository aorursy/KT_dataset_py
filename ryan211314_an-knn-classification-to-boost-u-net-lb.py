import numpy as np
import pandas as pd
from skimage.io import imread, imshow, imread_collection, concatenate_images
import os
from os.path import join
import glob
import cv2
%matplotlib inline
import matplotlib.pyplot as plt
from skimage.transform import resize

TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 3

train_ids = os.listdir(TRAIN_PATH)
test_ids = os.listdir(TEST_PATH)

train_image_paths = [glob.glob(join(TRAIN_PATH, train_id, "images", "*"))[0] for train_id in train_ids]
test_image_paths = [glob.glob(join(TEST_PATH, test_id, "images", "*"))[0] for test_id in test_ids]

from tqdm import tqdm


def get_image_finfo(image_paths):
    # complete img ,rgb mode
    full_img_list = []
    # just grey mode
    img_list = []
    
    # average value of gray pixels per image
    average_list = []
    
    # max Contour area value per image
    max_cnt_area = []
    
    # mean Contour area value per image
    average_cnt_area = []
    
    # how many Contour areas per image
    num_cnt = []
    
    #  width per image
    wid_list = []
    
    #  length per image
    len_list = []
    
    #  red per image
    r=[]
    
    #  green
    g=[]
    
    #  blue
    b=[]
    for case in tqdm(image_paths, total=len(image_paths)): 
        img = imread(case)[:,:,:IMG_CHANNELS]
        full_img_list.append(img)  
        r.append(np.average(img[:,:,0]))
        g.append(np.average(img[:,:,1]))
        b.append(np.average(img[:,:,2]))
        
        img = cv2.imread(case,cv2.IMREAD_GRAYSCALE)
        
        # in some cases, image background is bright and cell darker, there needs a inverse of pixel value
        if np.average(img) > 125:
            img = 255 - img   
        img_list.append(img)

        lenth = img.shape[0]
        len_list.append(lenth)
        width = img.shape[1]
        wid_list.append(width)
        average_list.append(np.average(img))
        
        # use opencv to find contour and get some stactistic data
        img = cv2.GaussianBlur(img, (3, 3), 1)
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

        _, cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        max_cnt_area.append(cv2.contourArea(cnts[0])/lenth/width)

        av = 0
        for i in cnts:
            av = av + cv2.contourArea(i)
        av = av/len(cnts)
        
        # since different pic has different size, we'd better normalise it 
        average_cnt_area.append(av/lenth/width)
        num_cnt.append(len(cnts))
        
    df = pd.DataFrame({'img':full_img_list,'max_area':max_cnt_area,'average_area':average_cnt_area,
                       'num_cnt':num_cnt,'average':average_list,'wid':wid_list,'len':len_list,
                       'r':r,'g':g,'b':b
                      }) 
    return df
df = get_image_finfo(train_image_paths)
# divide into 3 categories according to characters belong to form
FDIV = 3
# divide into 3 categories according to characters belong to color
CDIV = 3
from sklearn.cluster import KMeans 

# train seperately
input_x = np.array(df[['max_area','average_area','num_cnt','average','wid','len']])
  
fkmeans = KMeans(n_clusters = FDIV).fit(input_x) 

df['flabel'] = fkmeans.labels_

input_c = np.array(df[['r','g','b']])
  
ckmeans = KMeans(n_clusters = CDIV).fit(input_c) 

df['clabel'] = ckmeans.labels_

# and then make an combination
df['cflabel'] = FDIV *df['flabel']

df['cflabel'] = df['cflabel'] + df['clabel']

df['cflabel'].hist()
for t in range(FDIV*CDIV):
    print('for type=>'+str(t))
    fig,ax= plt.subplots(2,10,figsize=(32,5))
    
    n=0
    for i in range(2):
        for j in range(10):
            if n < len(df[df['cflabel']==t].index):
                sn = df[df['cflabel']==t].index[n]
                ax[i,j].imshow(df.img[sn])
                n = n+1
    plt.show()

test_df = get_image_finfo(test_image_paths)

input_x = np.array(test_df[['max_area','average_area','num_cnt','average','wid','len']])
  
test_df['flabel'] = fkmeans.predict(input_x)

input_c = np.array(test_df[['r','g','b']])
  
test_df['clabel'] = ckmeans.predict(input_c)

test_df['cflabel'] = FDIV *test_df['flabel']

test_df['cflabel'] = test_df['cflabel'] + test_df['clabel']

test_df['cflabel'].hist()
for t in range(FDIV*CDIV):
    print('for type=>'+str(t))
    
    fig,ax= plt.subplots(2,10,figsize=(32,5))
    n=0
    for i in range(2):
        for j in range(10):
            if n < len(test_df[test_df['cflabel']==t].index):
                sn = test_df[test_df['cflabel']==t].index[n]
                ax[i,j].imshow(test_df.img[sn])
                n = n+1
    plt.show()

fig,ax= plt.subplots(10,10,figsize=(32,32))
n=0
for i in range(10):
    for j in range(10):
        if n < len(test_df.index):
            
            ax[i,j].imshow(test_df.img[n])
            n = n+1
plt.show()
