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
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import MeanShift,KMeans
import os
path='/kaggle/input/imagedata/img.jpg'
img=cv2.imread(path)
plt.imshow(img)
img=cv2.resize(img,(100,100))
img.shape
new_img=img.reshape(-1,3)
new_img.shape
#model=MeanShift(n_jobs=-1)
model=KMeans(n_jobs=-1,n_clusters=3)
model.fit(new_img)
model.cluster_centers_
len(model.cluster_centers_)
model.predict([[1,1,1]])
new_img[0]
def color(x):
    if x==[0]:
        return [255,0,0]
    if x==[1]:
        return [0,255,0]
    if x==[2]:
        return [0,0,255]
    
for i in range(len(new_img)):
    new_img[i]=color(model.predict([new_img[i]]))

new_img[0]
new_img=new_img.reshape(100,100,3)
plt.imshow(new_img)
