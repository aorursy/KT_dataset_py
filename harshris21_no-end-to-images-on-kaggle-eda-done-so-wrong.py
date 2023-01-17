


from IPython.display import YouTubeVideo
YouTubeVideo("WPgJafGz4fg", height=500, width=700)
from IPython.display import YouTubeVideo
YouTubeVideo("1Q7ERNtLcvk", height=500, width=700)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; import seaborn as sns
plt.style.use('seaborn-whitegrid')
import openslide
import os
import cv2
import torch
train = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu
BASE_FOLDER='/kaggle/input/prostate-cancer-grade-assessment/'
train = pd.read_csv(BASE_FOLDER+"train.csv")
test = pd.read_csv(BASE_FOLDER+"test.csv")
sub = pd.read_csv(BASE_FOLDER+"sample_submission.csv")
plt.figure(figsize=(10, 7))
sns.countplot(train.data_provider)
plt.figure(figsize=(10, 7))
sns.countplot(train.isup_grade);
print('Total samples       :',train.shape[0])
for i in train.columns:
  print("Total No. of Unique values in Column {}  : {}".format(i,len(train[i].unique())))
  if len(train[i].unique()) <20:
    print(train[i].unique())
PATH = "../input/prostate-cancer-grade-assessment/"

df_train = pd.read_csv(f'{PATH}train.csv')
df_test = pd.read_csv(f'{PATH}test.csv')

df_train.head().style.set_caption('Quick Overview of train.csv')
print(f"Number of training data: {len(df_train)}\n")

print(f"Unique data_providers: {df_train.data_provider.unique()}\n")
print(f"Unique isup_grade: {df_train.isup_grade.unique()}\n")
print(f"Unique gleason_score: {df_train.gleason_score.unique()}\n")

print(f"Missing data:\n{df_train.isna().any()}\n")

masks = os.listdir(PATH + 'train_label_masks/')
images = os.listdir(PATH + 'train_images/')

df_masks = pd.Series(masks).to_frame()
df_masks.columns = ['mask_file_name']
df_masks['image_id'] = df_masks.mask_file_name.apply(lambda x: x.split('_')[0])
df_train = pd.merge(df_train, df_masks, on='image_id', how='outer')
del df_masks
print(f"There are {len(df_train[df_train.mask_file_name.isna()])} images without a mask.")
print(f"Train data shape before reduction: {len(df_train)}")
df_train_red = df_train[~df_train.mask_file_name.isna()]
print(f"Train data shape after reduction: {len(df_train_red)}")

no_masks = df_train[df_train.mask_file_name.isna()][['image_id']]
no_masks['Suspicious_because'] = 'No Mask'
df_train_red.groupby('isup_grade').gleason_score.unique().to_frame().style.set_caption('Mapping of ISUP Grade to Gleason Score')
df_train_red[(df_train_red.isup_grade == 2) & (df_train_red.gleason_score != '3+4')]
providers = df_train_red.data_provider.unique()

fig = plt.figure(figsize=(6,4))
ax = sns.countplot(x="isup_grade", hue="data_provider", data=df_train_red)
plt.title("ISUP Grade Count by Data Provider", fontsize=14)
plt.xlabel("ISUP Grade", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.show()


df_train_red["height"] = 0
df_train_red["width"] = 0
df_train_red[0] = 0
df_train_red[1] = 0
df_train_red[2] = 0
df_train_red[3] = 0
df_train_red[4] = 0
df_train_red[5] = 0

def get_image_data(row):
    biopsy = skimage.io.MultiImage(PATH + 'train_label_masks/' + row.image_id + '_mask.tiff')
    temp = biopsy[-1][:, :, 0]
    counts = pd.Series(temp.reshape(-1)).value_counts()
    row.height = temp.shape[0]
    row.width = temp.shape[1]
    row.update(counts)
    return row





import skimage.io

df_train_red = df_train_red.apply(lambda row: get_image_data(row), axis=1)
df_train_red['pixels'] = df_train_red.height * df_train_red.width
fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

"""
Inspired by something similiar I saw here https://www.kaggle.com/dhananjay3/panda-eda-all-you-need-to-know
"""
sns.scatterplot(data=df_train_red, x='width', y='height', marker='.',hue='data_provider', ax=ax1)
ax1.set_title("Image Sizes by Data Provider", fontsize=14)
ax1.set_xlabel("Image Width", fontsize=14)
ax1.set_ylabel("Image Height", fontsize=14)

sns.kdeplot(df_train_red[df_train_red.data_provider == 'karolinska'].pixels, label='karolinska', ax=ax2)
sns.kdeplot(df_train_red[df_train_red.data_provider == 'radboud'].pixels, label= 'radboud', ax=ax2)

ax2.set_title("Image Sizes by Data Provider", fontsize=14)
ax2.set_ylabel("Pixels per Image", fontsize=14)
plt.show()