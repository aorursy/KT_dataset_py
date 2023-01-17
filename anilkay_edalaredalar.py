# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

i=0

for dirname, _, filenames in os.walk('/kaggle/input'):

    break

    if i==4:

        break

    i=i+1    

    for filename in filenames:

        j=0

        print(os.path.join(dirname, filename))

        j=j+1

        if j==4:

            break

print("anıl")

# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/celeba-dataset/list_attr_celeba.csv")

data.head()
baldactive=data[["Attractive","Bald"]]

print(baldactive.corr())
lipsattack=data[["Attractive","Big_Lips"]]

print(lipsattack.corr())
youngattractive=data[["Attractive","Young"]]

print(youngattractive.corr())
yosmilhair=data["Young"]*data["Smiling"]*data["Black_Hair"]

prdata=pd.DataFrame({

    "attr":data["Attractive"],

    "youngsmilehair":yosmilhair

})

prdata.corr()
yosmilhair=data["Young"]*data["Black_Hair"]

prdata=pd.DataFrame({

    "attr":data["Attractive"],

    "youngsmilehair":yosmilhair

})

prdata.corr()
yosmilhair=data["Young"]*data["Big_Lips"]

prdata=pd.DataFrame({

    "attr":data["Attractive"],

    "youngsmilehair":yosmilhair

})

prdata.corr()
yosmilhair=data["Young"]*data["Big_Lips"]*data["Black_Hair"]

images=pd.DataFrame({

    "imgname":data["image_id"],

    "attr":data["Attractive"],

    "youngsmilehair":yosmilhair

})
imagesP=images[images["youngsmilehair"]==1]
%matplotlib inline

import skimage.io as imio

import matplotlib.pyplot as plt

first20=imagesP[0:20]["imgname"]
imgpath="/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/"

for img in first20:

    photo=imio.imread(imgpath+img)

    imio.imshow(photo)

    plt.show()
yosmilhair=data["Young"]+data["Big_Lips"]+data["Black_Hair"]

images=pd.DataFrame({

    "imgname":data["image_id"],

    "attr":data["Attractive"],

    "youngsmilehair":yosmilhair

})

imagesP=images[images["youngsmilehair"]==3]

first20=imagesP[0:20]["imgname"]

for img in first20:

    photo=imio.imread(imgpath+img)

    imio.imshow(photo)

    plt.show()
imagesP=images[images["youngsmilehair"]==3 & (images["attr"]==1)]

first20=imagesP[0:20]["imgname"]

for img in first20:

    photo=imio.imread(imgpath+img)

    imio.imshow(photo)

    plt.show()
yosmilhair=data["Attractive"]+data["Big_Nose"]

images=pd.DataFrame({

    "imgname":data["image_id"],

    "attr":data["Attractive"],

    "youngsmilehair":yosmilhair

})

imagesP=images[images["youngsmilehair"]==2]

first20=imagesP[0:20]["imgname"]

for img in first20:

    photo=imio.imread(imgpath+img)

    imio.imshow(photo)

    plt.show()