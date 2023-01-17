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

        #print(os.path.join(dirname, filename))

        pass

# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import cv2

import pandas as pd

import seaborn as sns
data = pd.read_csv('/kaggle/input/severstal-steel-defect-detection/train.csv')

print(data.head())
ImageId = []

ClassId = []

for i in data.ImageId_ClassId:

    ImageId.append(i.split('_')[0])

    ClassId.append(i.split('_')[1])

EncodedPixels =  list(data.EncodedPixels)

data2 = pd.DataFrame({'ImageId':ImageId,'ClassId':ClassId,'EncodedPixels':EncodedPixels})

data2.head()
data2.fillna(0,inplace=True)

data2['ClassId'] = data2['ClassId'].astype(int)

data2.head()

sns.countplot(data2.ClassId[data2.EncodedPixels!=0])

plt.show()
data2['ClassId'].dtype
Imagecount = []

for i in range(1,5):

    counter = 0

    for j in range(data2.shape[0]):

        if data2.EncodedPixels[j]!=0:

            if data2.ClassId[j]==i:

                counter+=1

    Imagecount.append(counter)

print(Imagecount)
def rleToMask(rleString,height,width):

    rows,cols = height,width

    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]

    rlePairs = np.array(rleNumbers).reshape(-1,2)

    img = np.zeros(rows*cols,dtype=np.uint8)

    for index,length in rlePairs:

        index -= 1

        img[index:index+length] = 255

    img = img.reshape(cols,rows)

    img = img.T

    return img
img = cv2.imread(os.path.join('/kaggle/input/severstal-steel-defect-detection/train_images',data2.ImageId[0]))

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

mask = rleToMask(data2.EncodedPixels[0],img.shape[0],img.shape[1])



masked_image = cv2.addWeighted(img,0.8,mask,1,1)

plt.imshow(masked_image,cmap='gray')

plt.show()
