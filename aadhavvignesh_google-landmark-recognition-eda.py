import pandas as pd
import numpy as np
from collections import Counter
import plotly.express as px
import matplotlib.pyplot as plt
import cv2
df = pd.read_csv('/kaggle/input/landmark-recognition-2020/train.csv')
df.head()
print("Number of training images:", len(df))
print("Number of landmarks:" ,df['landmark_id'].nunique())
landmark_counts = dict(Counter(df['landmark_id']))
landmark_dict = {'landmark_id': list(landmark_counts.keys()), 'count': list(landmark_counts.values())}

landmark_count_df = pd.DataFrame.from_dict(landmark_dict)
landmark_count_sorted = landmark_count_df.sort_values('count', ascending = False)
landmark_count_sorted.head(30)
fig_count = px.histogram(landmark_count_df, x = 'landmark_id', y = 'count')
fig_count.update_layout(
    title_text='Distribution of Landmarks',
    xaxis_title_text='Landmark ID',
    yaxis_title_text='Count'
)

fig_count.show()
BASE_DIR = '../input/landmark-recognition-2020'
TRAIN_DIR = BASE_DIR + '/train'

import os

filelist = []
for root, dirs, files in os.walk(TRAIN_DIR):
    for file in files:
        filelist.append(os.path.join(root,file))
len(filelist)
img_sizes = []

for img_path in filelist[:1000]:
    img = cv2.imread(img_path)
    img_sizes.append("{}x{}".format(img.shape[0], img.shape[1]))
size_counts = dict(Counter(img_sizes))
size_dict = {'size': list(size_counts.keys()), 'count': list(size_counts.values())}

size_df = pd.DataFrame.from_dict(size_dict)
size_sorted = size_df.sort_values('count', ascending = False)
size_sorted = size_sorted[:10]

fig_image_sizes = px.bar(size_sorted, x = 'size', y = 'count')
fig_image_sizes.update_layout(title = 'Image Sizes')
fig_image_sizes.show()
def retrieve_image(image_id):
    img = cv2.imread(os.path.join(os.path.join(BASE_DIR, 'train'), image_id[0], image_id[1], image_id[2], image_id + '.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_image_id(image_id):
    return df[df['landmark_id'] == image_id]['id'][:1].values[0]
fig, ax = plt.subplots(5, 2, figsize = (30, 30), dpi = 250)
ax = ax.flatten()

top_10_landmarks = landmark_count_sorted['landmark_id'][:10].values

for i in range(10):    
    ax[i].set_title(get_image_id(top_10_landmarks[i]))
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].imshow(retrieve_image(get_image_id(top_10_landmarks[i])))
fig.tight_layout()    
# plt.show()
fig, ax = plt.subplots(5, 2, figsize = (30, 30), dpi = 250)
ax = ax.flatten()
bottom_10_landmarks = landmark_count_sorted['landmark_id'][-10:].values

for i in range(10):
    ax[i].set_xticks([])
    ax[i].set_yticks([])    
    ax[i].imshow(retrieve_image(get_image_id(bottom_10_landmarks[i])))
    ax[i].set_title(get_image_id(bottom_10_landmarks[i]))
fig.tight_layout()
# plt.show()