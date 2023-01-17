# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import cv2

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_relationships.csv')

train.head()
train_list = os.listdir("../input/train")

len(train_list)
test_list = os.listdir("../input/test")

len(test_list)
def convert_image_to_array(image_dir):

    try:

        image = cv2.imread(image_dir)

        if image is not None :

            return list(image)

        else :

            return np.array([])

    except Exception as e:

        print(f"Error : {e}")

        return None
train_image_list = os.listdir("../input/train/"+train.p1[0])

train_image_list1 = os.listdir("../input/train/"+train.p2[0])



p1 = []

p2 = []

for image in train_image_list:

    p1.append(convert_image_to_array("../input/train/"+train.p1[0]+"/"+image))

for image in train_image_list1:

    p2.append(convert_image_to_array("../input/train/"+train.p2[0]+"/"+image))
plt.figure(figsize=(16,10))

for i in range(1,10):

    plt.subplot(2,5,i)

    plt.grid(False)

    plt.imshow(p1[i])

    #plt.xlabel(label_list[i])

plt.show()
plt.figure(figsize=(16,10))

for i in range(1,8):

    plt.subplot(2,5,i)

    plt.grid(False)

    plt.imshow(p2[i])

    #plt.xlabel(label_list[i])

plt.show()