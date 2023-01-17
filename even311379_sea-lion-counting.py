# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import glob

import os

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



train_data = pd.read_csv('../input/Train/train.csv')

train_imgs = sorted(glob.glob('../input/Train/*.jpg'), key=lambda name: int(os.path.basename(name)[:-4]))

train_dot_imgs = sorted(glob.glob('../input/TrainDotted/*.jpg'), key=lambda name: int(os.path.basename(name)[:-4]))



submission = pd.read_csv('../input/sample_submission.csv')





print(train_data.shape)

print('Number of Train Images: {:d}'.format(len(train_imgs)))

print('Number of Dotted-Train Images: {:d}'.format(len(train_dot_imgs)))







print(train_data.head(6))



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
index = 5

sl_counts = train_data.iloc[index]

print(sl_counts)



plt.figure()

sl_counts.plot(kind='bar', title='Count of Sea Lion Types')

plt.show()



print(train_imgs[index])

img = cv2.cvtColor(cv2.imread(train_imgs[index]), cv2.COLOR_BGR2RGB)

img_dot = cv2.cvtColor(cv2.imread(train_dot_imgs[index]), cv2.COLOR_BGR2RGB)



crop_img = img[200:2000, 2600:3500]

crop_img_dot = img_dot[200:2000, 2600:3500]



f, ax = plt.subplots(1,2,figsize=(16,8))

(ax1, ax2) = ax.flatten()



ax1.imshow(img)

ax2.imshow(img_dot)



plt.show()