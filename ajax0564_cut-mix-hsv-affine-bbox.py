# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
!ls /kaggle/input/save-cut-mix-image-box/wheat_cutmix
import matplotlib.pyplot as plt



img = cv2.imread('/kaggle/input/save-cut-mix-image-box/wheat_cutmix/image_100_1.jpg')

boxes = np.load('/kaggle/input/save-cut-mix-image-box/wheat_cutmix/box_100_1.npy')

fig, ax = plt.subplots(1, figsize=(8, 8))

for box in boxes:

        cv2.rectangle(img,

                  (int(box[1]), int(box[2])),

                  (int(box[3]), int(box[4])),

                  220, 3)



ax.imshow(img)

plt.show()
img = cv2.imread('/kaggle/input/save-cut-mix-image-box/wheat_cutmix/image_100_2.jpg')

boxes = np.load('/kaggle/input/save-cut-mix-image-box/wheat_cutmix/box_100_2.npy')

fig, ax = plt.subplots(1, figsize=(8, 8))

for box in boxes:

        cv2.rectangle(img,

                  (int(box[1]), int(box[2])),

                  (int(box[3]), int(box[4])),

                  220, 3)



ax.imshow(img)

plt.show()
img = cv2.imread('/kaggle/input/save-cut-mix-image-box/wheat_cutmix/image_100_3.jpg')

boxes = np.load('/kaggle/input/save-cut-mix-image-box/wheat_cutmix/box_100_3.npy')

fig, ax = plt.subplots(1, figsize=(8, 8))

for box in boxes:

        cv2.rectangle(img,

                  (int(box[1]), int(box[2])),

                  (int(box[3]), int(box[4])),

                  220, 3)



ax.imshow(img)

plt.show()
img = cv2.imread('/kaggle/input/save-cut-mix-image-box/wheat_cutmix/image_100_5.jpg')

boxes = np.load('/kaggle/input/save-cut-mix-image-box/wheat_cutmix/box_100_5.npy')

fig, ax = plt.subplots(1, figsize=(8, 8))

for box in boxes:

        cv2.rectangle(img,

                  (int(box[1]), int(box[2])),

                  (int(box[3]), int(box[4])),

                  220, 3)



ax.imshow(img)

plt.show()
img = cv2.imread('/kaggle/input/save-cut-mix-image-box/wheat_cutmix/image_100_6.jpg')

boxes = np.load('/kaggle/input/save-cut-mix-image-box/wheat_cutmix/box_100_6.npy')

fig, ax = plt.subplots(1, figsize=(8, 8))

for box in boxes:

        cv2.rectangle(img,

                  (int(box[1]), int(box[2])),

                  (int(box[3]), int(box[4])),

                  220, 3)



ax.imshow(img)

plt.show()