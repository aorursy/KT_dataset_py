import os



import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
BASE_DIR = '../input/landmark-recognition-2020'
!ls {BASE_DIR}
train_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))

train_df.head(10)
print(f'Total number of training images: {len(train_df)}')
print(f'Total number of landmarks in training dataset: {train_df["landmark_id"].nunique()}')
target_dist = train_df.groupby('landmark_id', as_index=False)['id'].count().sort_values('id', ascending=False).reset_index(drop=True)

target_dist = target_dist.rename(columns={'id':'count'})

target_dist
sns.set(rc={'figure.figsize':(11,8)})

sns.set(style="whitegrid")
ax = sns.distplot(train_df['landmark_id'].value_counts()[:50])

ax.set(xlabel='Landmark Counts', ylabel='Probability Density', title='Distribution of top 50 landmarks')

plt.show()
ax = sns.distplot(train_df['landmark_id'].value_counts()[51:])

ax.set(xlabel='Landmark Counts', ylabel='Probability Density')

plt.show()
def get_image(image_id):

    img = cv2.imread(os.path.join(os.path.join(BASE_DIR, 'train'), image_id[0], image_id[1], image_id[2], image_id + '.jpg'))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img



def get_image_id(landmark_id):

    return train_df[train_df['landmark_id'] == landmark_id]['id'][:1].values[0]
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 15))

ax = ax.flatten()

landmark_ids = target_dist['landmark_id'][:6].values



for i in range(6):

    ax[i].imshow(get_image(get_image_id(landmark_ids[i])))

    ax[i].grid(False)

plt.show()
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 15))

ax = ax.flatten()

landmark_ids = target_dist['landmark_id'][-6:].values



for i in range(6):

    ax[i].imshow(get_image(get_image_id(landmark_ids[i])))

    ax[i].grid(False)

plt.show()