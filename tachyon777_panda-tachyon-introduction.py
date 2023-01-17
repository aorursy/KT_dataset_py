# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os



# There are two ways to load the data from the PANDA dataset:

# Option 1: Load images using openslide

import openslide

# Option 2: Load images using skimage (requires that tifffile is installed)

import skimage.io

import random

import seaborn as sns

import cv2



# General packages

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import PIL

from IPython.display import Image, display



# Plotly for the interactive viewer (see last section)

import plotly.graph_objs as go
# 大元のディレクトリ

BASE_PATH = '../input/prostate-cancer-grade-assessment'



# trainingとmask画像データのディレクトリ

data_dir = f'{BASE_PATH}/train_images'

mask_dir = f'{BASE_PATH}/train_label_masks'





# train.csv,その他testとsubmissionのディレクトリ

train = pd.read_csv(f'{BASE_PATH}/train.csv').set_index('image_id')

test = pd.read_csv(f'{BASE_PATH}/test.csv')

submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
display(train.head())

print("Shape of training data :", train.shape)

print("unique data provider :", len(train.data_provider.unique()))

print("unique isup_grade(target) :", len(train.isup_grade.unique()))

print("unique gleason_score :", len(train.gleason_score.unique()))
#ISUPグレード5(悪性度の高い)の画像をピックアップ

res_5 = train.query("isup_grade == 5 and data_provider == 'radboud'")

res_5.head()
#ISUPグレード0(正常組織)の画像をピックアップ

res_0 = train.query("isup_grade == 0 and data_provider == 'radboud'")

res_0.head()
def display_image(image_id,x,y,level):

    isup_grade = train.loc[image_id]["isup_grade"]

    biopsy = openslide.OpenSlide(os.path.join(data_dir,image_id + '.tiff'))

    biopsy_mask = openslide.OpenSlide(os.path.join(mask_dir,image_id + '_mask.tiff'))

    print("画像情報")

    print("image_id",image_id)

    print("ISUPグレード：",isup_grade)

    print("画像サイズ:",biopsy.dimensions)

    zoom = ["高い","普通","低い"]

    print("拡大率:",zoom[level])

    

    region = biopsy.read_region((x,y), level, (512, 512))

    region_mask = biopsy_mask.read_region((x,y), level, (512, 512))

    

    f,ax = plt.subplots(1,2,figsize=(16,18))

    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

    ax[0].imshow(region)

    ax[0].axis('off')

    ax[0].set_title(f"ID: {image_id}\nISUP: {isup_grade}")

    ax[1].imshow(np.asarray(region_mask)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)

    ax[1].axis('off')

    ax[1].set_title("mask_image")



    plt.show() 
images = ["007433133235efc27a39f11df6940829", #grade0

          "01642d24ac5520681d6a20f6c42dc4fe", #grade0

          "02577ddcd838f2559936453b6071dc17", #grade0

          "00928370e2dfeb8a507667ef1d4efcbb", #grade5

          "00c15b23b30a5ba061358d9641118904", #grade5

          "00c52cb4db1c7a5811a8f070a910c038"  #grade5

         ]
#一番遠い視野での画像

display_image("007433133235efc27a39f11df6940829",2000,8000,2)
#同画像を拡大したもの

display_image("007433133235efc27a39f11df6940829",4000,11000,1)

#更に上皮組織を拡大したもの

display_image("007433133235efc27a39f11df6940829",4300,12500,0)
#遠くから

display_image("00928370e2dfeb8a507667ef1d4efcbb",2000,8000,2)

#赤いマスクがされている部分について拡大

display_image("00928370e2dfeb8a507667ef1d4efcbb",5000,8000,1)

#更に拡大

display_image("00928370e2dfeb8a507667ef1d4efcbb",6200,8700,0)
#正常細胞例2

display_image("01642d24ac5520681d6a20f6c42dc4fe",5000,5000,2)
#緑マスク部にめがけて拡大。

display_image("01642d24ac5520681d6a20f6c42dc4fe",5000,7000,1)

#更に拡大

display_image("01642d24ac5520681d6a20f6c42dc4fe",6000,7300,0)
#ISUPグレード5(悪性度高)細胞例2

display_image("00928370e2dfeb8a507667ef1d4efcbb",2500,8000,2)
#悪性部に拡大

display_image("00928370e2dfeb8a507667ef1d4efcbb",4500,8500,1)

#さらに拡大

display_image("00928370e2dfeb8a507667ef1d4efcbb",5500,8700,0)
#また、同画像のオレンジ(グリソンスコア4)の部分についても見てみる。

display_image("00928370e2dfeb8a507667ef1d4efcbb",4500,12000,1)

#さらに拡大

display_image("00928370e2dfeb8a507667ef1d4efcbb",5200,12800,0)
#データを提供してる研究所は以下の2つ。

train["data_provider"].unique()
def plot_ct(data):

    left = [i for i in range(len(data))]

    height = [i[1] for i in data]

    labels = [i[0] for i in data]

    plt.bar(left, height, width=0.5,linewidth=2, tick_label=labels)

    plt.title("data_provider")

    plt.ylabel("count")

    plt.xlabel("provider")

    plt.show()

    



data = [

    ['karolinska',(train["data_provider"] =='karolinska').sum()],

    ['radboud',(train["data_provider"] =='radboud').sum()]

]





plot_ct(data)

print("施設ごとのデータ提供割合：")

for i,j in data:

    print(i,":",round(j/len(train)*100,2),"%")
def plot_isup(data):

    left = [i for i in range(len(data))]

    height = [i[1] for i in data]

    labels = [i[0] for i in data]

    plt.bar(left, height, width=0.5,linewidth=2, tick_label=labels)

    plt.title("isup_grade")

    plt.ylabel("count")

    plt.xlabel("isup_grade")

    plt.show()

    



data = [[i,(train["isup_grade"] == i).sum()] for i in range(6)]



plot_isup(data)

print("それぞれのISUPグレードの割合：")

for i,j in data:

    print("グレード",i,":",round(j/len(train)*100,2),"%")
print('karolinskaのみ')

data = [[i,((train["isup_grade"] == i) & (train["data_provider"] =='karolinska')).sum()] for i in range(6)]



plot_isup(data)

print("それぞれのISUPグレードの割合：")

for i,j in data:

    print("グレード",i,":",round(j/len(train)*100,2),"%")
print('radboudのみ')

data = [[i,((train["isup_grade"] == i) & (train["data_provider"] =='radboud')).sum()] for i in range(6)]



plot_isup(data)

print("それぞれのISUPグレードの割合：")

for i,j in data:

    print("グレード",i,":",round(j/len(train)*100,2),"%")
train["gleason_score"].unique()
def plot_gleason(data):

    left = [i for i in range(len(data))]

    height = [i[1] for i in data]

    labels = ["nega" if i[0] == "negative" else i[0] for i in data]

    plt.bar(left, height, width=0.5,linewidth=2, tick_label=labels)

    plt.title("gleason_score")

    plt.ylabel("count")

    plt.xlabel("gleason_score")

    plt.show()

    



data = [[i,(train["gleason_score"] == i).sum()] for i in train["gleason_score"].unique()]

plot_gleason(data)

print("それぞれのグリソン分類の割合：")

for i,j in data:

    print(i,":",round(j/len(train)*100,2),"%")
for i in ["karolinska","radboud"]:

    res = train[train["data_provider"] == i]

    print("0+0 in ",i,":",any("0+0" == i for i in res["gleason_score"]))

    print("negative in ",i,":",any("negative" == i for i in res["gleason_score"]))

print('karolinskaのみ')



data = [[i,((train["gleason_score"] == i) & (train["data_provider"] =='karolinska') ).sum()] for i in train["gleason_score"].unique()]

plot_gleason(data)

print("それぞれのグリソン分類の割合：")

for i,j in data:

    print(i,":",round(j/len(train)*100,2),"%")
print('radboudのみ')



data = [[i,((train["gleason_score"] == i) & (train["data_provider"] =='radboud') ).sum()] for i in train["gleason_score"].unique()]

plot_gleason(data)

print("それぞれのグリソン分類の割合：")

for i,j in data:

    print(i,":",round(j/len(train)*100,2),"%")