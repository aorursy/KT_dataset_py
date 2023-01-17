# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
import pandas as pd
import matplotlib.pyplot as plt

#load train csv file
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

print('Examples with melanoma')

#Get 10 malignant training image_names.  
imgs = train.loc[train.target==1].sample(10).image_name.values

#Show images 
plt.figure(figsize=(20,8))
for i,k in enumerate(imgs):
    img = cv2.imread('../input/siim-isic-melanoma-classification/jpeg/train/%s.jpg'%k)
    #change color
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.subplot(2,5,i+1)
    plt.axis('off')
    plt.imshow(img)
plt.show()

#Show Examples WITHOUT melanoma
print('Examples WITHOUT melanoma')

#Get 10 benign lesion.
imgs = train.loc[train.target==0].sample(10).image_name.values

plt.figure(figsize=(20,8))
for i,k in enumerate(imgs):
    img = cv2.imread('../input/siim-isic-melanoma-classification/jpeg/train/%s.jpg'%k)
    #change color
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.subplot(2,5,i+1)
    plt.axis('off')
    plt.imshow(img)
plt.show()