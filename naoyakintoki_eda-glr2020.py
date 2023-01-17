import glob
from PIL import Image
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
input_dir = '../input/landmark-retrieval-2020/'
glob.glob(input_dir+'*')
train_images = sorted(glob.glob(input_dir+"train/*/*/*/*.jpg"))
test_images = sorted(glob.glob(input_dir+"test/*/*/*/*.jpg"))
index_images = sorted(glob.glob(input_dir+"index/*/*/*/*.jpg"))

print(f'train : {len(train_images)} \
        test : {len(test_images)} \
        index : {len(index_images)}')
def plot_sample_images(images, nrows=2, ncols=4):
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, int(nrows*2.5)))
    for i, ax in enumerate(axes.flatten()):
        img = plt.imread(images[i])
        ax.imshow(img)
    plt.show()
plot_sample_images(train_images)
plot_sample_images(test_images)
plot_sample_images(index_images)
train_df = pd.read_csv(input_dir+'train.csv')
train_df
train_df.landmark_id.unique().shape
train_df.landmark_id.value_counts()
fig, axis = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(train_df['landmark_id'], order=train_df['landmark_id'].value_counts().index[:20], ax=axis[0])
sns.countplot(train_df['landmark_id'], order=train_df['landmark_id'].value_counts().index[-20:], ax=axis[1])
axis[0].tick_params(axis='x', labelrotation=90)
axis[1].tick_params(axis='x', labelrotation=90)
axis[1].set(ylim=(0, 10))
plt.show()
def trans_id_to_path(id):
    path = f'{input_dir}/train/{id[0]}/{id[1]}/{id[2]}/{id}.jpg'
    return path

train_df['path'] = train_df['id'].map(trans_id_to_path)
train_df.head(5)
train_df.to_pickle(f'train_path.zip')
def display_images_per_id(df, id, n=2):
    nrows = n
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, int(nrows*2.5)))
    cnt = len(df[df['landmark_id']==id])
    print(f'Landmark ID = {id}')
    for i in range(nrows):
        for j in range(ncols):
            img = plt.imread(df[df['landmark_id']==id].path.iloc[random.randint(0, cnt - 1)])
            axes[i, j].imshow(img)
    plt.show()
i = random.randint(0, len(train_df)-1)
display_images_per_id(train_df, id=train_df.landmark_id[i])
i = random.randint(0, len(train_df)-1)
display_images_per_id(train_df, id=train_df.landmark_id[i])
i = random.randint(0, len(train_df)-1)
display_images_per_id(train_df, id=train_df.landmark_id[i])