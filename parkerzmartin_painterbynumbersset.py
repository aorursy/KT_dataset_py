# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import cv2

import shutil



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    break

    

    for filename in filenames:

        os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.



# returns file path from the train.zip

def imgPath(num):

    path = "../input/painter-by-numbers/train/" + num

    return path



# prints out image from filename column

def printImg(num):

    path = imgPath(num)

    print(path)

    plt.figure(figsize=(12,12))

    plt.subplot(1,2,1)

    img = cv2.imread(path)

    imgplot = plt.imshow(img)

    

    plt.show()
# import train info along with removing art without any style

pbn = pd.read_csv("../input/painter-by-numbers/train_info.csv")

pbn = pbn.fillna(np.nan)

pbn.drop(labels = ["title","genre","date","artist"], axis=1, inplace=True)

pbn = pbn.dropna(how='any',axis=0)
pbn.head()
stylesDict = {}



for index, row in pbn.iterrows():

    if row["style"] in stylesDict:

        stylesDict[row["style"]] = stylesDict[row["style"]] + 1

    else:

        stylesDict[row["style"]] = 1
for x in stylesDict:

    if(stylesDict[x] > 1000):

        print(x,": ",stylesDict[x])
# removing unneeded images from the data in a better way

stylesToKeep = []

for x in stylesDict:

    if stylesDict[x] > 1000:

        stylesToKeep.append(x)

for index, row in pbn.iterrows():

    if not row["style"] in stylesToKeep:

        pbn.drop(index, inplace=True)
pbn
pbn.describe()
# pbn.to_csv('cleaned.csv', index = False)

# shutil.os.mkdir('/kaggle/output/images')

# path = '/kaggle/output/images/'

# for x in stylesToKeep:

#     shutil.os.mkdir(path+x)

    

# shutil.move('/kaggle/output/cleaned.csv',/kaggle/output/images/cleaned.csv)
# prints out an image



# %pylab inline

# import matplotlib.pyplot as plt

# import matplotlib.image as mpimg



# img=imread('../input/painter-by-numbers/train_2/2000.jpg')

# imgplot = plt.imshow(img)

# plt.show()