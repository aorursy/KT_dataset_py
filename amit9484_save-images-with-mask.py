# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import matplotlib.pylab as plt

from keras.preprocessing.image import load_img

import os

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





stringpath = r"../input/pku-autonomous-driving/"

# Any results you write to the current directory are saved as output.

train_data   = pd.read_csv('../input/pku-autonomous-driving/train.csv')



train_imagesfolder = os.listdir(stringpath + r"/train_images") # dir is your directory path

trainimagesfilecount = len(train_imagesfolder)



train_masksfolder = os.listdir(stringpath +  r"train_masks") # dir is your directory path

trainmasksfilecount = len(train_imagesfolder)



traindata   = pd.read_csv(stringpath + r"train.csv")





def CreateMaskImages(imageName):



    trainimage = cv2.imread(stringpath  + "/train_images/" + imageName + '.jpg')

    imagemask = cv2.imread(stringpath + "/train_masks/" + imageName + ".jpg",0)

    try:

        imagemaskinv = cv2.bitwise_not(imagemask)

        res = cv2.bitwise_and(trainimage,trainimage,mask = imagemaskinv)

        plt.imshow(imagemask)

        cv2.imwrite("MaskTrain/" + imageName + ".jpg", res)

    except:

        print("exception for image" + imageName)

        cv2.imwrite("MaskTrain/" + imageName + ".jpg", trainimage)

        

        

for i in range(len(traindata)):

  ImageName = traindata.loc[i, "ImageId"]

  print(ImageName)

  #CreateMaskImages(ImageName)