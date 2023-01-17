import os
import gc
import json
import math
import cv2
import PIL
import re
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
#from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
%matplotlib inline
#from keras.preprocessing import image
import glob
import tensorflow.keras.applications.densenet as dense
from kaggle_datasets import KaggleDatasets
import seaborn as sns
sns.set_style('whitegrid')

import missingno as msno

from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
tf.__version__
train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

print('Train: ', train.shape)
print("Test:", test.shape)
train.head()
test.head()
msno.matrix(train, );
vc = train.groupby("benign_malignant")["diagnosis"].value_counts().unstack()[train["diagnosis"].value_counts().sort_values().index]
display(vc)
vc.iplot(kind='bar', yTitle='Percentage', 
          linecolor='black', 
          opacity=0.7,
          theme='pearl',
          bargap=0.5,
          gridcolor='white',
          barmode = 'stack',
          title='Distribution of the Target column in the training set')
vc = train.groupby("benign_malignant")["sex"].value_counts(normalize=True).unstack()
vc.iplot(kind='bar', yTitle='Percentage', 
          linecolor='black', 
          opacity=0.7,
          theme='pearl',
          bargap=0.5,
          gridcolor='white',
          barmode = 'stack',
          title='Target vs Gender')
plt.figure(figsize=(12,5))

sns.distplot(train.loc[train['sex'] == 'female', 'age_approx'], label = 'Benign')

sns.distplot(train.loc[train['sex'] == 'male', 'age_approx'], label = 'Malignant')

scipy.stats.ttest_ind(train.loc[train['sex'] == 'female', 'age_approx'], train.loc[train['sex'] == 'male', 'age_approx'], nan_policy='omit')
vc = train.groupby("age_approx")["benign_malignant"].value_counts().unstack()
vc.iplot(kind='bar', yTitle='Percentage', 
          linecolor='black', 
          opacity=0.7,
          theme='pearl',
          bargap=0.2,
          gridcolor='white',
          barmode = 'stack',
          title='Age vs Gender')
vc = train.groupby("age_approx")["benign_malignant"].value_counts(normalize=True).unstack()
vc.iplot(kind='bar', yTitle='Percentage', 
          linecolor='black', 
          opacity=0.7,
          theme='pearl',
          bargap=0.2,
          gridcolor='white',
          barmode = 'stack',
          title='Age vs Gender, normalized')
plt.figure(figsize=(12,5))

sns.distplot(train.loc[train['target'] == 0, 'age_approx'], label = 'Benign')

sns.distplot(train.loc[train['target'] == 1, 'age_approx'], label = 'Malignant')

scipy.stats.ttest_ind(train.loc[train['target'] == 0, 'age_approx'], train.loc[train['target'] == 1, 'age_approx'], nan_policy='omit')
vc = train["diagnosis"].value_counts()[::-1]
vc[vc.index != "unknown"].plot.barh()

vc = train.groupby("anatom_site_general_challenge")["benign_malignant"].value_counts().unstack()
vc.iplot(kind='bar', yTitle='Percentage', 
          linecolor='black', 
          opacity=0.7,
          theme='pearl',
          bargap=0.2,
          gridcolor='white',
          barmode = 'stack',
          title='Target vs Gender')
vc = train.groupby("anatom_site_general_challenge")["benign_malignant"].value_counts(normalize=True).unstack()
vc.iplot(kind='bar', yTitle='Percentage', 
          linecolor='black', 
          opacity=0.7,
          theme='pearl',
          bargap=0.2,
          gridcolor='white',
          barmode = 'stack',
          title='Target vs Gender')





def display_training_curves(training, validation, title, subplot):
  if subplot%10==1: # set up the subplots on the first call
    plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
    plt.tight_layout()
  ax = plt.subplot(subplot)
  ax.set_facecolor('#F8F8F8')
  ax.plot(training)
  ax.plot(validation)
  ax.set_title('model '+ title)
  ax.set_ylabel(title)
  ax.set_xlabel('epoch')
  ax.legend(['train', 'valid.'])

def grid_display(list_of_images, no_of_columns=2, figsize=(15,15), title = False):
    num_images = len(list_of_images)
    no_of_rows = int(num_images / no_of_columns)
    fig, axes = plt.subplots(no_of_rows,no_of_columns, figsize=figsize)
    if no_of_rows == 1:
        list_axes = []
        list_axes.append(axes)
        axes = list_axes
    
    idx = 0
    idy = 0
    
    for i, img in enumerate(list_of_images):
        axes[idy][idx].imshow(img)
        axes[idy][idx].axis('off')
        if title:
            axes[idy][idx].set_title(title[i])
            
        if idx < no_of_columns - 1:
            idx+=1
        else:
            idx=0
            idy+=1
    fig.tight_layout()
    return fig
image_list = train[train['target'] == 0].sample(16)['image_name']
image_all=[]
for image_id in image_list:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    image_all.append(img)
#show_images(image_all, cols=1)
fig = grid_display(image_all, 4, (15,15), title = range(16))
image_list = train[train['target'] == 1].sample(16)['image_name']
image_all=[]
for image_id in image_list:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    image_all.append(img)
fig = grid_display(image_all, 4, (15,15), title = range(16))
image_list = train[(train['anatom_site_general_challenge'] == 'torso') & (train['target'] == 1)].sample(16)['image_name']
image_all=[]
for image_id in image_list:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    image_all.append(img)
fig = grid_display(image_all, 4, (15,15), title = range(16))
image_list = train[(train['anatom_site_general_challenge'] == 'lower extremity') & (train["target"] == 1)].sample(16)['image_name']
image_all=[]
for image_id in image_list:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    image_all.append(img)
fig = grid_display(image_all, 4, (15,15), title = range(16))
image_list = train[(train['anatom_site_general_challenge'] == 'upper extremity') & (train["target"] == 1)].sample(16)['image_name']
image_all=[]
for image_id in image_list:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    image_all.append(img)
fig = grid_display(image_all, 4, (15,15), title = range(16))
image_list = train[(train['anatom_site_general_challenge'] == 'head/neck') & (train["target"] == 1)].sample(16)['image_name']
image_all=[]
for image_id in image_list:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    image_all.append(img)
fig = grid_display(image_all, 4, (15,15), title = range(16))
image_list = train[(train['anatom_site_general_challenge'] == 'palms/soles') & (train["target"] == 1)].sample(4)['image_name']
image_all=[]
for image_id in image_list:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    image_all.append(img)
fig = grid_display(image_all, 4, (15,15), title = range(16))
image_list = train[train['diagnosis'] == 'seborrheic keratosis'].sample(16)['image_name']
image_all=[]
for image_id in image_list:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    image_all.append(img)
fig = grid_display(image_all, 4, (15,15), title = range(16))
image_list = train[train['diagnosis'] == 'lentigo NOS'].sample(16)['image_name']
image_all=[]
for image_id in image_list:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    image_all.append(img)
fig = grid_display(image_all, 4, (15,15), title = range(16))
image_list = train[train['diagnosis'] == 'lichenoid keratosis'].sample(16)['image_name']
image_all=[]
for image_id in image_list:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    image_all.append(img)
fig = grid_display(image_all, 4, (15,15), title = range(16))
image_list = train[train['diagnosis'] == 'solar lentigo'].sample(4)['image_name']
image_all=[]
for image_id in image_list:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    image_all.append(img)
fig = grid_display(image_all, 4, (15,15), title = range(4))
image_list = train[train['diagnosis'] == 'atypical melanocytic proliferation'].sample(1)['image_name']
image_all=[]
for image_id in image_list:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))

plt.imshow(img)
plt.axis('off');
image_list = train[train['diagnosis'] == 'cafe-au-lait macule'].sample(1)['image_name']
image_all=[]
for image_id in image_list:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    image_all.append(img)
plt.imshow(img)
plt.axis('off');
arr = [15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,90.0]
image_all=[]
titles = ['At Age 15.0','At Age 20.0','At Age 25.0','At Age 30.0','At Age 35.0','At Age 40.0'
          ,'At Age 45.0','At Age 50.0','At Age 55.0','At Age 60.0','At Age 65.0','At Age 70.0'
          ,'At Age 75.0','At Age 80.0','At Age 85.0','At Age 90.0']
for i in arr:
    image_list = train[(train['age_approx'] == i) & (train["target"] == 1)].sample()['image_name']
    for image_id in image_list:
        image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
        img = np.array(Image.open(image_file))
        image_all.append(img)
fig = grid_display(image_all, 4, (15,15), title = titles)
benign_images = train[train["target"] == 0].sample(10)["image_name"]
cancer_images = train[train["target"] == 1].sample(10)["image_name"]

benign_image_arr = []
cancer_image_arr = []

for image_id in benign_images:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    benign_image_arr.append(img)
    
for image_id in cancer_images:
    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 
    img = np.array(Image.open(image_file))
    cancer_image_arr.append(img)
reds.mean()/255, greens.mean()/255, blues.mean()/255
reds = np.hstack([v[:, :, 0].ravel() for v in benign_image_arr])
greens = np.hstack([v[:, :, 1].ravel() for v in benign_image_arr])
blues = np.hstack([v[:, :, 2].ravel() for v in benign_image_arr])

plt.figure(figsize=(15, 8))
_ = plt.hist(reds, bins=256, color='red', alpha=0.5)
_ = plt.hist(greens, bins=256, color='green', alpha=0.5)
_ = plt.hist(blues, bins=256, color='blue', alpha=0.5)

_ = plt.xlabel('Intensity Value')
_ = plt.ylabel('Count')
_ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])

print("R: {:.2f}, G: {:2f}, B: {:2f}".format(reds.mean(), greens.mean(), blues.mean()))

plt.show()
reds = np.hstack([v[:, :, 0].ravel() for v in cancer_image_arr])
greens = np.hstack([v[:, :, 1].ravel() for v in cancer_image_arr])
blues = np.hstack([v[:, :, 2].ravel() for v in cancer_image_arr])

plt.figure(figsize=(15, 8))
_ = plt.hist(reds, bins=256, color='red', alpha=0.5)
_ = plt.hist(greens, bins=256, color='green', alpha=0.5)
_ = plt.hist(blues, bins=256, color='blue', alpha=0.5)

_ = plt.xlabel('Intensity Value')
_ = plt.ylabel('Count')
_ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])

print("R: {:.2f}, G: {:2f}, B: {:2f}".format(reds.mean(), greens.mean(), blues.mean()))

plt.show()
# img = Image.open(train_path + 'ISIC_2637011.jpg')

# light = transforms.Compose([
#     transforms.RandomErasing()
#     ])


# fig, axes = plt.subplots(1,2, figsize=(12, 6))
# axes[0].imshow(img)
# axes[1].imshow(transforms.RandomErasing()(np.array(img)))
# # axes[1].imshow(Cutout(scale=(0.05, 0.007), value=(0, 0))(np.array(img)))

# axes[0].axis('off')
# axes[1].axis('off')