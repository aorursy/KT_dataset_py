# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import cv2

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Show some image data

imageList = os.listdir("../input/train_images/")[:12]

print(imageList)

plt.figure(figsize=(27,27))

for index, imageFile in enumerate(imageList):

    filePath = "../input/train_images/" + imageFile

    image = cv2.imread(filePath)

    plt.subplot(4,3,index+1)

    plt.title("#{}".format(index+1))

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))



plt.show()

train_df = pd.read_csv("../input/train.csv")

train_df.head()
unicode_df = pd.read_csv("../input/unicode_translation.csv")

unicode_df