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
from tqdm import tqdm

import cv2 as cv2

import matplotlib.pyplot as plt

import seaborn as sns
traincsv=pd.read_csv('/kaggle/input/severstal-steel-defect-detection/train.csv')

trainImgsPath='/kaggle/input/severstal-steel-defect-detection/train_images'

testImgsPath='/kaggle/input/severstal-steel-defect-detection/test_images'
traincsv['ImageId']=traincsv['ImageId_ClassId'].str.split('_',expand=True)[0]

traincsv['ClassID']=traincsv['ImageId_ClassId'].str.split('_',expand=True)[1]

traincsv['HasLabel']=(~traincsv['EncodedPixels'].isna()).astype(int)
def getRandomImgsFrom(Path,noOfImgs):

    imgs=[]

    folder=os.listdir(Path)

    for img in tqdm(folder):

        if(len(imgs)<noOfImgs):

            imgPath=os.path.join(Path,img)

            tempImg=cv2.imread(imgPath)

            if tempImg is not None:

                imgs.append(tempImg)

        else:

            break

    return imgs
def plotImgs(imgs):

    fig=plt.figure(figsize=(20,20))

    rows,cols=5,4

    for i in range(0, cols*rows):        

        fig.add_subplot(rows, cols, i+1)

        plt.imshow(imgs[i],aspect='equal')

    plt.show() 
def mask2rle(mask):

    flat=mask.T.flatten()    

    padded=np.concatenate([[0],flat,[0]])    

    runs=np.where(padded[1:]!=padded[:-1])[0]   

    runs+=1

    runs[1::2]-=runs[0::2]

    print(runs)    
mask2rle(trainImgs[15])
trainImgs=getRandomImgsFrom(trainImgsPath,20)

plotImgs(trainImgs)
dummydf=traincsv.groupby('ImageId').sum()

dummydf.sort_values('HasLabel', ascending=False,inplace=True)

dummydf.reset_index(inplace=True)

dummydf.columns=['ImageId','NoOfLables']

dummydf.info()