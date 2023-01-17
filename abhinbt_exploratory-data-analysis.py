

import numpy as np # Importing numpy

import pandas as pd # Importing pandas



# Input available at "../input/" directory.

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#reading the CSV file from the directory

train_df=pd.read_csv("../input/train.csv")
#checking more information about dataset

train_df.info()
train_df.head()
train_df.tail()
#Total Nulls in the 'EncodedPixels'

train_df['EncodedPixels'].isnull().sum()
import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

import tqdm

from PIL import Image

%matplotlib inline
trainfiles=os.listdir("../input/train_images/")
#Length of trainfiles

len(trainfiles)



image_size=[]

for image_id in trainfiles:

    img=Image.open("../input/train_images/"+image_id)

    width,height=img.size

    image_size.append((width,height))

   
image_size_df=pd.DataFrame(image_size,columns=["width","height"])

image_size_df.head()
names = ['width', 'height']

plt.hist([image_size_df["width"], image_size_df["height"]],label=names)

plt.legend()
testfiles=os.listdir("../input/test_images/")
len(testfiles)
image_size=[]

for image_id in testfiles:

    img=Image.open("../input/test_images/"+image_id)

    width,height=img.size

    image_size.append((width,height))
test_image_size_df=pd.DataFrame(image_size,columns=["width","height"])

test_image_size_df.head()
names = ['width', 'height']

plt.hist([test_image_size_df["width"], test_image_size_df["height"]],label=names)

plt.legend()