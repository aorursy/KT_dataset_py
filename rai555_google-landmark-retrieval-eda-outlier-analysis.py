import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import cv2

import glob

from tqdm import tqdm

import random

import torch

import os
def seed_everything(seed_value):

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True

#        torch.backends.cudnn.benchmark = True



seed = 42

seed_everything(seed)
train = pd.read_csv("/kaggle/input/landmark-retrieval-2020/train.csv")

train.head()
print('There are {} rows and {} columns in train.csv'.format(train.shape[0], train.shape[1]))
train_files = glob.glob('../input/landmark-retrieval-2020/train/*/*/*/*')

test_files = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')

index_files = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')
print('Number of files in train folder: {}'.format(len(train_files)))

print('Number of files in test folder : {}'.format(len(test_files)))

print('Number of files in index folder: {}'.format(len(index_files)))

print('Number of unique landmarks: {}'.format(train['landmark_id'].nunique()))
# Plot

sns.set_style('whitegrid', {'axes.grid' : False})

fig = plt.figure()

fig.set_size_inches(18, 12)



for i in range(12):

    plt.subplot(3, 4, i+1)

    img = cv2.imread(train_files[i])

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



plt.tight_layout()
# Function to get dimensions of images

def get_dims(file_list, n=5000):

    files = random.sample(file_list, n)

    image_dimensions = []

    

    for f in tqdm(files):

        img = cv2.imread(f)

        image_dimensions.append(img.shape)

        

    width = [dimensions[0] for dimensions in image_dimensions] 

    height = [dimensions[1] for dimensions in image_dimensions] 

    df_dims = pd.DataFrame(width, columns =['width']) 

    df_dims['height'] = height    

    return df_dims



# Get dimensions of train, test and index files

train_dims = get_dims(train_files)

test_dims = get_dims(test_files, n=1129)

index_dims = get_dims(index_files)
# Plot

sns.set_style('whitegrid')

fig, axs = plt.subplots(1,3, figsize=(24, 6))

axs = axs.ravel()



dims_data = [index_dims, train_dims, test_dims]

titles = ['Image Dimensions - Index', 'Image Dimensions - Train', 'Image Dimensions - Test']



for i in range(3):

    dims_data[i].plot.scatter(x = "width", y = "height", ax=axs[i], color='#20A387', alpha=0.6, s=75, edgecolors='#187864')

    axs[i].set_title(titles[i], fontsize=14)
train_grouped = train['landmark_id'].value_counts().rename_axis('landmark_id').reset_index(name='image_count').sort_values(by='image_count', ascending=False) 



print ('Maximum number of images for a given landmark_id: {}'.format(train_grouped.image_count.max()))

print ('Minimum number of images for a given landmark_id: {}'.format(train_grouped.image_count.min()))

print ('Mean number of images for a given landmark_id   : {}'.format((round(train_grouped.image_count.mean()))))

print ('Median number of images for a given landmark_id : {}'.format(train_grouped.image_count.median()))
sns.set_style('whitegrid')

fig, ax = plt.subplots(1, 2)

fig.set_size_inches(20, 6)



sns.lineplot(x=train_grouped.index.values, y='image_count', data=train_grouped, color='#20A387', linewidth=3, ax=ax[0])

sns.lineplot(x=train_grouped.index.values, y='image_count', data=train_grouped, color='#20A387', linewidth=3, ax=ax[1])



ax[0].set_title('Images Per Landmark ID', fontsize=14)

ax[1].set_title('Images Per Landmark ID on a Log Scale', fontsize=14)

ax[0].set(xlabel='Landmarks')

ax[1].set(xlabel='Landmarks')

ax[1].set(yscale="log")



plt.show()
fig, ax = plt.subplots(1, 2)

fig.set_size_inches(20, 6)



sns.countplot(train['landmark_id'], order=train['landmark_id'].value_counts().index[:25], color='#20A387', ax=ax[0], alpha=0.75)

sns.countplot(train['landmark_id'], order=train['landmark_id'].value_counts().index[81288:],color='#20A387', ax=ax[1], alpha=0.75)



ax[0].tick_params(axis='x', labelrotation=70)

ax[1].tick_params(axis='x', labelrotation=70)

ax[0].set_title('Landmark ids with the Highest Image Count', fontsize=14)

ax[1].set_title('Landmark ids with the Least Image Count', fontsize=14)

ax[1].set(ylim=(0, 100))





plt.show()
path='../input/landmark-retrieval-2020/train/'



def plot_landmarks(landmark_id, df, width=12, height=18, nrows=1, ncolumns=2):    

    landmark_df = df[df['landmark_id']==landmark_id]

    sns.set_style('whitegrid', {'axes.grid' : False})

    fig = plt.figure()

    fig.set_size_inches(width, height)

    

    pos = 0

    for i, row in landmark_df.iterrows(): 

        image_id =  row['id']

        plt.subplot(nrows, ncolumns, pos+1)

        img = cv2.imread(path+'/'+image_id[0]+'/'+image_id[1]+'/'+image_id[2]+'/'+image_id+'.jpg')

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        pos += 1

    

    plt.tight_layout()
plot_landmarks(landmark_id=183115, df=train)
plot_landmarks(landmark_id=197219, df=train)
plot_landmarks(landmark_id=63266, df=train)
plot_landmarks(landmark_id=138982, df=train[train['landmark_id']==138982][0:12], width=18, height=24, nrows=4, ncolumns=3)
plot_landmarks(landmark_id=126637, df=train[train['landmark_id']==126637][0:9], width=18, height=15, nrows=3, ncolumns=3)