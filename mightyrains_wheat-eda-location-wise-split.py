# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import imageio

import shutil

import random

from tqdm.auto import tqdm

import matplotlib.pyplot as plt
img_path = '../input/global-wheat-detection/train/'



def display_img(image_idx):

    img = imageio.imread(f'{img_path}/{image_idx}.jpg')

    plt.imshow(img)

    



def display_all_locations():



    for location in set(df['combined_src']):

        loc_mask = df['combined_src'] == location



        plt.figure(figsize=(20, 15))

        plt.suptitle(location)

        r, c = 10, 10

        for i in range(r*c):

            rand_idx = random.randint(0, df[loc_mask].shape[0]-1)

            plt.subplot(r, c, i+1)

            display_img(df[loc_mask].iloc[rand_idx, 0])



        # break
orig_df = pd.read_csv('../input/global-wheat-detection/train.csv')

orig_df.head()
orig_df['source'].hist()

orig_df['source'].value_counts()
orig_df['combined_src'] = orig_df['source'].apply(lambda x: x.split('_')[0])

orig_df['combined_src'].hist()

orig_df['combined_src'].value_counts()
df = orig_df.copy()



bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

for i, column in enumerate(['x', 'y', 'w', 'h']):

    df[column] = bboxs[:,i]

    

df.drop(columns=['bbox'], inplace=True)

df['x_center'] = df['x'] + df['w']/2

df['y_center'] = df['y'] + df['h']/2

df['classes'] = 0



index = list(set(df.image_id))



df.head()
from sklearn.model_selection import train_test_split, GroupKFold

folds_info = []



for fold_idx, (train_idx, valid_idx) in enumerate(GroupKFold(4).split(df, groups=df.combined_src)):

    train_df = df.iloc[train_idx].reset_index(drop=True)

    valid_df = df.iloc[valid_idx].reset_index(drop=True)



    plt.figure()

    train_df['combined_src'].hist()

    valid_df['combined_src'].hist()

    name = "valid_on_" + "_".join(list(set(valid_df['combined_src'])))

    plt.title(name)



    print(name, train_idx.shape, valid_idx.shape, f"ratio: {valid_idx.shape[0] / train_idx.shape[0] * 100:.2f}")

    folds_info.append(

        [train_df, valid_df, name]

    )

    # break
display_all_locations()
def dump(mdf, labels_path, images_path):



    for img_idx, mini in tqdm(mdf.groupby('image_id')):

        img_loc = f'{img_path}/{img_idx}.jpg'



        shutil.copy(

            img_loc,

            f'{images_path}/{img_idx}.jpg'

        )

        

        with open(f'{labels_path}/{img_idx}.txt', 'w+') as f:

            row = mini[['classes','x_center','y_center','w','h']].astype(float).values

            row = row/1024

            row = row.astype(str)

            for j in range(len(row)):

                text = ' '.join(row[j])

                f.write(text)

                f.write("\n")

        # break
def save(train_df, valid_df, savename):

    outpath = "/kaggle/working/files/"

    train_path_labels = f'{outpath}/train/labels/'

    train_path_images = f'{outpath}/train/images/'

    valid_path_labels = f'{outpath}/valid/labels/'

    valid_path_images = f'{outpath}/valid/images/'



    for p in [train_path_labels, train_path_images, valid_path_labels, valid_path_images]:

        os.makedirs(p, exist_ok=True)

        

    dump(train_df, train_path_labels, train_path_images)

    dump(valid_df, valid_path_labels, valid_path_images)

    

    os.system(f'zip -rmq {savename}.zip files')
!ls
%%time



for (train_df, valid_df, savename) in folds_info:

    save(train_df, valid_df, savename)
!ls -hl