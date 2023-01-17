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
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')
data.head()
print(f'there are {data.image_id.nunique()}  unique images in the data')
print(f'Total number of samples{data.shape[0]}')
sample_img = plt.imread('/kaggle/input/global-wheat-detection/train/b6ab77fd7.jpg')
plt.imshow(sample_img)
x_min,y_min,width,height = list(map(int,[377.0, 504.0, 74.0, 160.0]))
plt.imshow(sample_img[x_min:x_min+width, y_min:y_min+height,:])
x_min,y_min,width,height = list(map(int,[834.0, 222.0, 56.0, 36.0]))
plt.imshow(sample_img[x_min:x_min+width, y_min:y_min+height,:])
x_min,y_min,width,height = list(map(int,[226.0, 548.0, 130.0, 58.0]))
plt.imshow(sample_img[x_min:x_min+width, y_min:y_min+height,:])
data.isnull().sum()
len(os.listdir('/kaggle/input/global-wheat-detection/train'))
data.image_id.unique()
train_image_ids = [s.strip('.jpg') for s in os.listdir('/kaggle/input/global-wheat-detection/train')]
train_image_ids
len(train_image_ids) - data.image_id.nunique()
image_ids_without_wheat = list(set(train_image_ids).difference(set(data.image_id.unique())))
len(image_ids_without_wheat)
sample_image_without_wheat = plt.imread('/kaggle/input/global-wheat-detection/train/'+image_ids_without_wheat[0]+'.jpg')
plt.imshow(sample_image_without_wheat)
for i in range(1,10):
    plt.subplot(3,3,i)  
    plt.imshow(plt.imread('/kaggle/input/global-wheat-detection/train/'+image_ids_without_wheat[i-1]+'.jpg'))
import cv2
cv2.imread()