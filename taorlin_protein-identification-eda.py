import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from skimage.io import imread
import cv2
from skimage.transform import resize
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator

import keras
import keras.backend as K

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/train.csv', index_col=0)
df.head()
images=[imread('../input/train/'+x+'_green.png', as_gray=True) for x in df.index]
print(images[0].shape)
df['Image']=images
thresholded=df.iloc[:20, 1].apply(lambda x: cv2.thr)
from skimage.filters import try_all_threshold
fig, ax = try_all_threshold(df.iloc[0, 1], figsize=(10, 8), verbose=False)
plt.show()
plt.imshow(df.iloc[0, 1])
plt.show()
shape=[10, 5]
fig=plt.figure(figsize=(20, 35))
for i in range(shape[0]*shape[1]):
    sub=plt.subplot(shape[0], shape[1], i+1)
    plt.imshow(df.iloc[i, 1])
print('Example Images')
plt.show()
# Clustering
df['Hist']=df['Image'].apply(lambda x: np.histogram(x, 128)[0])
df.head()
from sklearn.cluster import KMeans
clusters=10
kmeans=KMeans(clusters)
df['Cluster']=kmeans.fit_predict([np.array(x) for x in df['Hist'].values])
df.head()
shape=[10, 5]
fig=plt.figure(figsize=(20, 35))
for i in range(shape[0]*shape[1]):
    sub=plt.subplot(shape[0], shape[1], i+1)
    sub.set_title(df.iloc[i, 3])
    plt.imshow(df.iloc[i, 1])
print('Example Images')
plt.show()
plt.figure(figsize=(25, 12))
df['Classes']=df['Target'].apply(lambda x: np.array([int(t) for t in x.split(' ')]))
for i in range(clusters):
    cluster=df.loc[df['Cluster']==i]
    classes=cluster['Classes']
    al=[]
    for k in classes:
        al.extend(list(k))
    al=np.array(al)
    hist=np.histogram(al, 28)
    sub=plt.subplot(2, 5, i+1)
    plt.bar(range(28), hist[0])
plt.show()
