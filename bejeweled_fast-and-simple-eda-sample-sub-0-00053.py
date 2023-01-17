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
import cv2



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')

sns.set(font_scale=1)
df_train = pd.read_csv("../input/train.csv")

labels = pd.read_csv("../input/labels.csv")

sample_sub = pd.read_csv("../input/sample_submission.csv")
df_train.head()
labels.head()
sample_sub.head()
df_train.shape, labels.shape, sample_sub.shape
df_train.groupby("attribute_ids").size().sort_values()[::-1][:10]
df_train.groupby("attribute_ids").size().sort_values()[::-1][:10].hist(bins=10)
print("train size is {}".format(len(os.listdir("../input/train/"))))

print("test size is {}".format(len(os.listdir("../input/test/"))))
c = 1

plt.figure(figsize=[16,16])

for img_name in os.listdir("../input/train/")[:16]:

    img = cv2.imread("../input/train/{}".format(img_name))[...,[2,1,0]]

    plt.subplot(4,4,c)

    plt.imshow(img)

    plt.title("train image {}".format(c))

    c += 1

plt.show();
c = 1

plt.figure(figsize=[16,16])

for img_name in os.listdir("../input/test/")[:16]:

    img = cv2.imread("../input/test/{}".format(img_name))[...,[2,1,0]]

    plt.subplot(4,4,c)

    plt.imshow(img)

    plt.title("test image {}".format(c))

    c += 1

plt.show();
sample_sub.to_csv("submission.csv", index=False)