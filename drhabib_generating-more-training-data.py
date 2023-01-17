from PIL import Image,ImageDraw,ImageFont

import numpy as np

from pathlib2 import Path

from functools import partial

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from scipy.ndimage import filters

from tqdm import tqdm

import os

import random
SIZE = 224

HEIGHT = 137

WIDTH = 236

PATH =  '../input/bengali-synth-data/'



df = pd.read_csv(PATH + 'BengaliWordList_439.csv').sample(frac=1).reset_index(drop=True).iloc[:, 0].dropna()
#random font that i found on internet, 

#one can add more fonts to mimic hand writing 



font_list= [ImageFont.truetype(PATH + "kalpurush.ttf",100), 

              ImageFont.truetype(PATH + "Atma-Light.ttf",100), 

              ImageFont.truetype(PATH + "Galada-Regular.ttf",100), 

              ImageFont.truetype(PATH + "Mina-Regular.ttf",100)]



def DrawBengli(txt,font):

    image = np.zeros(shape=(HEIGHT,WIDTH),dtype=np.uint8)

    x = Image.fromarray(image)

    draw = ImageDraw.Draw(x)

    draw.text((10,10),txt,(255),font=random.choice(font))

    p = np.array(x)

    return p



#lafoss kernel 128 x 128

#https://www.kaggle.com/iafoss/image-preprocessing-128x128

def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size=SIZE, pad=20):

    ymin,ymax,xmin,xmax = bbox(img0[:,5:-5] > 80)

    xmin = xmin - 8 if(xmin > 8) else 0

    ymin = ymin - 5 if(ymin > 5) else 0

    xmax = xmax + 8 if(xmax < WIDTH - 8) else WIDTH

    ymax = ymax + 5 if(ymax < HEIGHT - 5) else HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    img[img < 30] = 0

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + pad

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return Image.fromarray(img).resize((224, 224), resample=Image.BILINEAR).convert('RGB')



    

def guas_filter(img):

    brush = np.random.uniform(1, 4)

    return Image.fromarray(filters.gaussian_filter(img, brush))



def get_random_part(ts, max_ln = 4):

    #getting random parts of the word

    len_wd = len(ts)

    try:

        return ts[np.random.randint(0, len_wd//2): np.random.randint(len_wd//2, len_wd)][:np.random.randint(1, max_ln)]

    except:

        return ts[:np.random.randint(1, 4)]
def resize_one(indx, FOLDER = 'img_synt_224_thin', test=False, fonts = font_list):

    wd_orig = df.iloc[indx]

    wd = get_random_part(wd_orig)

    img = crop_resize(DrawBengli(wd, fonts))

    if test:

        #print (wd_orig)

        return guas_filter(img)

    else:

        guas_filter(img).save(f'{FOLDER}/cust_1_{indx}.png')
fig=plt.figure(figsize=(8, 8))

columns = 7

rows = 7

for i in range(1, columns*rows +1):

    img = resize_one(i + 10, test=True)

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

plt.show()