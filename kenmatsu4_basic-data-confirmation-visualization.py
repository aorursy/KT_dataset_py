# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import math, sys, functools, os, codecs, gc, time
import importlib  
from glob import glob
import numpy as np
import numpy.random as rd
import pandas as pd
import scipy as sp
from scipy import stats as st
from datetime import  datetime as dt
from collections import Counter
from pathlib import Path

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
plt.rcParams["patch.force_edgecolor"] = True
#rc('text', usetex=True)
from IPython.display import display # Allows the use of display() for DataFrames
import seaborn as sns
sns.set(style="whitegrid", palette="muted", color_codes=True)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
red = sns.xkcd_rgb["light red"]
green = sns.xkcd_rgb["medium green"]
blue = sns.xkcd_rgb["denim blue"]

def current_time():
    return dt.strftime(dt.now(),'%Y-%m-%d %H:%M:%S')

#カラム内の文字数。デフォルトは50
pd.set_option("display.max_colwidth", 50)

#行数
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)   

%matplotlib inline
%config InlineBackend.figure_format='retina'

import  pickle
def unpickle(filename):
    with open(filename, 'rb') as fo:
        p = pickle.load(fo)
    return p

def to_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, -1)
print("[input]")
!ls ../input/
print("="*10)
print("[input/train]")
!ls ../input/train
print("="*10)
print("[input/test]")
!ls ../input/test
####################################################
# Parameter settings & data loading

# path
TRAIN_IMG_PATH = Path("../input/train/images/")
TRAIN_MSK_PATH = Path("../input/train/masks/")
TEST_IMG_PATH = Path("../input/test/images/")

# data frames
df_train = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
df_depths = pd.read_csv("../input/depths.csv", index_col="id")

# image file list
train_images = [str(p) for p in TRAIN_IMG_PATH.glob("*.png")]
mask_images  = [str(p) for p in TRAIN_MSK_PATH.glob("*.png")]
test_images  = [str(p) for p in TEST_IMG_PATH.glob("*.png")]
mask_images[0]
# sample of depth.csv
print("DataFrame of depth.csv")
display(df_depths.loc[[i.split("/")[-1].replace(".png","") for i in test_images[:10]]])
# sample of train.csv
print("DataFrame of train.csv")
display(df_train.head(10))
# number of images
print(f"The number of train images: {len(train_images)}")
print(f"The number of test images: {len(test_images)}")
# image shape
shape = plt.imread(train_images[0]).shape
print(f"The image shape is width:{shape[0]}, height:{shape[1]}, channel:{shape[2]}")
print("X axis means horizontal direction, Y axis is vertical direction (deepness), 3 channels contain same values, so  actually 1channel grey scale data.")
def draw_train_data(train_ids, start_pos = 0, n_col = 5, n_row = 10, 
                    font_size=10, mask_thresh_lower=0, mask_thresh_upper=1):
    n_img = n_col * n_row
    train_ids = train_ids[start_pos:] #train_images[start_pos:start_pos+n_img]
    
    plt.figure(figsize=(16, 3.5*n_row))
    cnt = 0
    for i, img_id in enumerate(train_ids):
        #print(f"\r i={i}", end="")
        #img_id = img.split("/")[-1].replace(".png","")
        #print(img_id, img, mk)
        img = f'../input/train/images/{img_id}.png'
        mk  = f'../input/train/masks/{img_id}.png'
        y_train = plt.imread(mk)
        mask_ratio  = y_train.sum()/np.prod(y_train.shape)
        
        # check range of mask ratio
        if mask_ratio < mask_thresh_lower or mask_thresh_upper < mask_ratio:
            continue
        
        x_train = plt.imread(img)
        depth = df_depths.loc[img_id].z
        color_depth = x_train.sum()/np.prod(x_train.shape)
        
        plt.subplot(n_row, n_col, cnt+1)
        plt.imshow(x_train.mean(axis=2), cmap="Spectral")
        plt.imshow(y_train, cmap="Greys", alpha=0.4)
        plt.title("ID:{0}, depth={1}, ratio={2:0.1f}%".format(img_id, depth, mask_ratio*100), fontsize=font_size)

        plt.grid(False)
        plt.axis('off')
        
        if cnt+1 >= n_img:
            break
        cnt += 1

    #plt.tight_layout()
    plt.subplots_adjust(top=1.0, bottom=0.0, 
                        left=0.0, right=1.0, 
                        hspace=0.1, wspace=0.05)
    plt.show()
draw_train_data(df_train.index)
draw_train_data(df_train.index, mask_thresh_lower=0.0001, mask_thresh_upper=0.2)
draw_train_data(df_train.index, mask_thresh_lower=0.2, mask_thresh_upper=0.5)
draw_train_data(df_train.index, mask_thresh_lower=0.5, mask_thresh_upper=0.8)
draw_train_data(df_train.index, mask_thresh_lower=0.8, mask_thresh_upper=1)
draw_train_data(df_train.index, mask_thresh_lower=0, mask_thresh_upper=0)
# no mask data of mask_ratio 1.0
draw_train_data(df_train.index, mask_thresh_lower=1, mask_thresh_upper=1)
n_col = 5
n_row = 10
n_img = n_col * n_row
start_pos = 50

plt.figure(figsize=(16, 3.5*n_row))
for i, img in enumerate(test_images[start_pos:start_pos+n_img]):
    print(f"\r i={i}", end="")
    img_id = img.split("/")[-1].replace(".png","")
    #print(img_id, img, mk)
    plt.subplot(n_row, n_col, i+1)
    x_train = plt.imread(img)
    plt.imshow(x_train.mean(axis=2), cmap="Spectral")
    
    depth = df_depths.loc[img_id].z
    plt.title(f"ID:{img_id}, depth={depth}", fontsize=8)
    
    plt.grid(False)
    plt.axis('off')
    
plt.subplots_adjust(top=1.0, bottom=0.0, 
                    left=0.0, right=1.0, 
                    hspace=0.05, wspace=0.05)
plt.show()
img_id_list = []
mask_ratio_list = []
color_depth_list = []
std_list = []
for i, (img, mk) in enumerate(zip(train_images, mask_images)):
    print(f"\r i={i}", end="")
    img_id = img.split("/")[-1].replace(".png","")
    #print(img_id, img, mk)
    x_train = plt.imread(img)
    y_train = plt.imread(mk)
    depth = df_depths.loc[img_id].z
    color_depth = x_train.sum()/np.prod(x_train.shape)
    mask_ratio  = y_train.sum()/np.prod(y_train.shape)
    img_id_list.append(img_id)
    mask_ratio_list.append(mask_ratio)
    color_depth_list.append(color_depth)
    std_list.append(x_train.std())
    
df_mask_ratio = pd.DataFrame({"id":img_id_list, "mask_ratio": mask_ratio_list}).set_index("id")
df_mask_ratio = df_mask_ratio.join(df_depths, on="id", how='inner')
df_mask_ratio = pd.DataFrame({"id":img_id_list, 
                              "mask_ratio": mask_ratio_list, 
                              "color_depth":color_depth_list,
                              "std_val": std_list
                             }).set_index("id")
df_mask_ratio = df_mask_ratio.join(df_depths, on="id", how='inner').rename(columns={"z":"depth"})
df_mask_ratio.head()
df_mask_ratio.mask_ratio.hist(bins=20)
plt.show()
# neighbor of 0
df_mask_ratio.mask_ratio.hist(bins=np.arange(0, 0.1, 0.005))
plt.xlim(0, 0.1)
plt.show()
df_depths.z.hist(bins=20)
plt.title("Histgram of depth")
plt.show()

bins = 20
alpha = 0.6
df_depths.loc[df_train.index].z.hist(bins=bins, alpha=alpha, density=True, label='TRAIN')
df_depths.loc[[i for i in df_depths.index if i not in df_train.index]].z.hist(bins=bins, alpha=alpha, density=True, label='TEST')
plt.title("Histgram of depth")
plt.legend(loc="best")
g = sns.jointplot(x="mask_ratio", y="depth", data=df_mask_ratio, kind="reg", color="b", 
                  xlim=(-0.05, 1.05), ylim=(0, 1000), height=8, marginal_kws=dict(bins=15, rug=True))
g.fig.suptitle("mask_ratio vs depth")
plt.show()
# xlim=(-0.05, 1.05), ylim=(0, 1000), 
g = sns.jointplot(x="mask_ratio", y="color_depth", data=df_mask_ratio, kind="reg", color="g", 
                  height=8, marginal_kws=dict(bins=15, rug=True))
g.fig.suptitle("mask_ratio vs color_depth")
plt.show()
# xlim=(-0.05, 1.05), ylim=(0, 1000), 
g = sns.jointplot(x="depth", y="color_depth", data=df_mask_ratio, kind="reg", color="m", 
                  height=8, marginal_kws=dict(bins=15, rug=True))
g.fig.suptitle("depth vs color_depth")
plt.show()
g = sns.jointplot(x="mask_ratio", y="std_val", data=df_mask_ratio, kind="reg", color="grey", 
                  height=8, marginal_kws=dict(bins=15, rug=True))
g.fig.suptitle("mask_ratio vs std_val")
plt.show()
g = sns.jointplot(x="color_depth", y="std_val", data=df_mask_ratio, kind="reg", color="orange", 
                  height=8, marginal_kws=dict(bins=15, rug=True))
g.fig.suptitle("color_depth vs std_val")
plt.show()
train_no_mask_ratio = (df_mask_ratio.mask_ratio==0).sum() / df_mask_ratio.shape[0]
print("{0:0.1f}% of the train data are no-mask.".format(train_no_mask_ratio*100))
