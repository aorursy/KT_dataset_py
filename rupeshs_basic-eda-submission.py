# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("../input/train.csv")
train_df.info()
train_df.head()
#Total Nans

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
__import__('pandas').read_csv('../input/sample_submission.csv',converters={'EncodedPixels':lambda e:''}).to_csv('submission.csv',index=False)