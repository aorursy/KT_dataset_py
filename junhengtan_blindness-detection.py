import numpy as np
import pandas as pd
import seaborn as sns
import os
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
train_path = os.path.join('..','input/train_images/')
df_train = pd.read_csv("../input/train.csv")
df_train['path'] = df_train['id_code'].map(lambda x: os.path.join(train_path,'{}.png'.format(x)))
df_train = df_train.drop(columns=['id_code'])
df_train.head()
#df_test = pd.read_csv("../input/train.csv")
sns.countplot(df_train['diagnosis'])
def subtract_median_bg_image(im):
    k = np.max(im.shape)//20*2+1
    bg = cv2.medianBlur(im, k)
    return cv2.addWeighted (im, 4, bg, -4, 128)

def subtract_gaussian_bg_image(im):
    k = np.max(im.shape)/10
    bg = cv2.GaussianBlur(im ,(0,0) ,k)
    return cv2.addWeighted (im, 4, bg, -4, 128)
fig, ax = plt.subplots(6, 5, figsize=(30, 15))
ax = ax.flatten()
for i in range(30):
    path = df_train['path'][i]
    img = cv2.imread(path)
    img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(512,512))
    ax[i].imshow(img)
    ax[i].set_title(df_train['diagnosis'][i])
    
plt.show()
fig, ax = plt.subplots(10, 5, figsize=(20, 20))
ax = ax.flatten()
for i in range(50):
    path = df_train['path'][i]
    img = cv2.imread(path)
    img = cv2.resize(img,(512,512))
    img = subtract_median_bg_image(img)
    ax[i].imshow(img)
    ax[i].set_title(df_train['diagnosis'][i])
    
plt.show()

fig, ax = plt.subplots(10, 5, figsize=(20, 20))
ax = ax.flatten()
for i in range(50):
    path = df_train['path'][i]
    img = cv2.imread(path)
    img  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(512,512))
    img = subtract_gaussian_bg_image(img)
    
    ax[i].imshow(img)
    ax[i].set_title(df_train['diagnosis'][i])
    
plt.show()
