import numpy as np 

import pandas as pd 

import cv2

from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (16,9)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
TRAIN_PATH = '/kaggle/input/train.csv'

TRAIN_IMAGES = '/kaggle/input/train_images/'

TEST_IMAGES = '/kaggle/input/test_images/'

UNICODE_PATH = '/kaggle/input/unicode_translation.csv'
data = pd.read_csv(TRAIN_PATH)

uni_trans = pd.read_csv(UNICODE_PATH)
data.head()
import re
sample = data.iloc[500]



labels = sample.labels

labels = re.findall(r"U\+.{4} \d+ \d+ \d+ \d+",labels)
img = cv2.imread(TRAIN_IMAGES + f'{sample.image_id}.jpg')
symbol_idx = 0 

x_idx = 1

y_idx = 2

w_idx = 3

h_idx = 4

for label in labels:

    values = label.split()

    x = int(values[x_idx])

    y = int(values[y_idx])

    w = int(values[w_idx])

    h = int(values[h_idx])

    

    cv2.rectangle(img,(x,y), (x+w,y+h),(0,0,255), 10)

    
plt.imshow(img)