import os



import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt #for plotting

import seaborn as sns            #vizualisation

DIRE = '../input/landmark-recognition-2020'
!ls {DIRE}
df = pd.read_csv(os.path.join(DIRE, 'train.csv'))

df.head(10)
print(f'Total number of training images: {len(df)}')

print(f'Total number of landmarks in training dataset: {df["landmark_id"].nunique()}')

target_dist = df.groupby('landmark_id', as_index=False)['id'].count().sort_values('id', ascending=False).reset_index(drop=True)

target_dist = target_dist.rename(columns={'id':'count'})

target_dist
ad = sns.distplot(df['landmark_id'].value_counts()[:50])

ad.set(xlabel='Landmark Counts', ylabel='Probability Density', title='Distribution of top 50 landmarks')

plt.show()
ad = sns.distplot(df['landmark_id'].value_counts()[51:])

ad.set(xlabel='Landmark Counts', ylabel='Probability Density')

plt.show()
def get_image(image_id):

    img = cv2.imread(os.path.join(os.path.join(DIRE, 'train'), image_id[0], image_id[1], image_id[2], image_id + '.jpg'))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img



def get_image_id(landmark_id):

    return df[df['landmark_id'] == landmark_id]['id'][:1].values[0]
fig, ad = plt.subplots(nrows=2, ncols=3, figsize=(30, 15))

ad = ad.flatten()

landmark_ids = target_dist['landmark_id'][:6].values



for i in range(6):

    ad[i].imshow(get_image(get_image_id(landmark_ids[i])))

    ad[i].grid(False)

plt.show()

fig, ad = plt.subplots(nrows=2, ncols=3, figsize=(30, 15))

ad = ad.flatten()

landmark_ids = target_dist['landmark_id'][-6:].values



for i in range(6):

    ad[i].imshow(get_image(get_image_id(landmark_ids[i])))

    ad[i].grid(False)

plt.show()