import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

import os

from skimage.io import imread

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

dsb_data_dir = os.path.join('..', 'input')

stage_label = 'stage1'
all_images = glob(os.path.join(dsb_data_dir, 'stage1_*', '*', '*', '*.png'))

img_df = pd.DataFrame({'path': all_images})

img_id = lambda in_path: in_path.split('/')[-3]

img_type = lambda in_path: in_path.split('/')[-2]

img_group = lambda in_path: in_path.split('/')[-4].split('_')[1]

img_stage = lambda in_path: in_path.split('/')[-4].split('_')[0]

img_df['ImageId'] = img_df['path'].map(img_id)

img_df['ImageType'] = img_df['path'].map(img_type)

img_df['TrainingSplit'] = img_df['path'].map(img_group)

img_df['Stage'] = img_df['path'].map(img_stage)

# we don't want any masks

img_df = img_df.query('ImageType=="images"').drop(['ImageType'],1)

img_df.sample(2)
%%time

img_df['images'] = img_df['path'].map(imread)

img_df.drop(['path'],1, inplace = True)

img_df.sample(1)
color_features_names = ['Gray', 'Red', 'Green', 'Blue', 'Red-Green',  'Red-Green-Sd']

def create_color_features(in_df):

    in_df['Red'] = in_df['images'].map(lambda x: np.mean(x[:,:,0]))

    in_df['Green'] = in_df['images'].map(lambda x: np.mean(x[:,:,1]))

    in_df['Blue'] = in_df['images'].map(lambda x: np.mean(x[:,:,2]))

    in_df['Gray'] = in_df['images'].map(lambda x: np.mean(x))

    in_df['Red-Green'] = in_df['images'].map(lambda x: np.mean(x[:,:,0]-x[:,:,1]))

    in_df['Red-Green-Sd'] = in_df['images'].map(lambda x: np.std(x[:,:,0]-x[:,:,1]))

    return in_df



img_df = create_color_features(img_df)

sns.pairplot(img_df[color_features_names+['TrainingSplit']], 

             hue = 'TrainingSplit')
from sklearn.cluster import KMeans

from string import ascii_lowercase



def create_color_cluster(in_df, cluster_maker = None, cluster_count = 3):

    if cluster_maker is None:

        cluster_maker = KMeans(cluster_count)

        cluster_maker.fit(in_df[['Green', 'Red-Green', 'Red-Green-Sd']])

        

    in_df['cluster-id'] = np.argmin(

        cluster_maker.transform(in_df[['Green', 'Red-Green', 'Red-Green-Sd']]),

        -1)

    in_df['cluster-id'] = in_df['cluster-id'].map(lambda x: ascii_lowercase[x])

    return in_df, cluster_maker



img_df, train_cluster_maker = create_color_cluster(img_df, cluster_count=4)

sns.pairplot(img_df,

             vars = ['Green', 'Red-Green', 'Red-Green-Sd'], 

             hue = 'cluster-id')
n_img = 3

grouper = img_df.groupby(['cluster-id', 'TrainingSplit'])

fig, m_axs = plt.subplots(n_img, len(grouper), 

                          figsize = (20, 4))

for (c_group, clus_group), c_ims in zip(grouper, 

                                     m_axs.T):

    c_ims[0].set_title('Group: {}\nSplit: {}'.format(*c_group))

    for (_, clus_row), c_im in zip(clus_group.sample(n_img, replace = True).iterrows(), c_ims):

        c_im.imshow(clus_row['images'])

        c_im.axis('off')

fig.savefig('messy_overview.png')
tiny_img_df = grouper.apply(lambda x: x.sample(n_img if n_img<x.shape[0] else x.shape[0])

                           ).reset_index(drop=True).drop(color_features_names, 1).sort_values(['cluster-id', 'TrainingSplit'])

print(tiny_img_df.shape[0], 'images to experiment with')

tiny_img_df.sample(2)
def show_test_img(in_df, in_col):

    plt_cols = tiny_img_df.shape[0]//4

    fig, m_axs = plt.subplots(4, plt_cols, figsize = (12, 12))

    for c_ax, (_, c_row) in zip(m_axs.flatten(), in_df.iterrows()):

        c_ax.imshow(c_row[in_col])

        c_ax.axis('off')

        c_ax.set_title('K:{cluster-id} T:{TrainingSplit}'.format(**c_row))

show_test_img(tiny_img_df, 'images')
import cv2

grid_size = 8

def rgb_clahe(in_rgb_img): 

    bgr = in_rgb_img[:,:,[2,1,0]] # flip r and b

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size,grid_size))

    lab[:,:,0] = clahe.apply(lab[:,:,0])

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return bgr[:,:,[2,1,0]]
tiny_img_df['clahe_lab'] = tiny_img_df['images'].map(rgb_clahe)
show_test_img(tiny_img_df, 'clahe_lab')
def rgb_clahe_justl(in_rgb_img): 

    bgr = in_rgb_img[:,:,[2,1,0]] # flip r and b

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size,grid_size))

    return clahe.apply(lab[:,:,0])

tiny_img_df['clahe_justl'] = tiny_img_df['images'].map(rgb_clahe_justl)
show_test_img(tiny_img_df, 'clahe_justl')
tiny_img_df['clahe_justl_flip'] = tiny_img_df['clahe_justl'].map(lambda x: 255-x if x.mean()>127 else x)
show_test_img(tiny_img_df, 'clahe_justl_flip')