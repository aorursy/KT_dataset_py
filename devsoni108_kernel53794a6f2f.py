# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

from PIL import Image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
dogs=[]

for dirname, _, filenames in os.walk('/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/dogs'):

    for filename in filenames:

        full_path=os.path.join(dirname, filename)

        img=cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)

        img=cv2.resize(img,(128,128))

        dogs.append(img)
cats=[]

for dirname, _, filenames in os.walk('/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/cats'):

    for filename in filenames:

        full_path=os.path.join(dirname, filename)

        img=cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)

        img=cv2.resize(img,(128,128))

        cats.append(img)
dogs_img=np.concatenate(dogs)

cats_img=np.concatenate(cats)
dogs_df=pd.DataFrame(dogs_img)

cats_df=pd.DataFrame(cats_img)
dogs_df['label']=1

cats_df['label']=0
all_images=pd.concat([dogs_df,cats_df])
all_images.head()
all_images.tail()
all_images.shape
features=all_images.drop('label',axis=1)

target=all_images['label']
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(features,target)