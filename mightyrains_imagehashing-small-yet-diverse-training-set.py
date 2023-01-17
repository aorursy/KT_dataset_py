!pip install imagehash
# You can modify these to suit your needs and resources.



NUM_SAMPLES_TO_COMPARE = 1000  # we random sample these many images to choose from.

NUM_SAMPLES_SAVE_PER_LANDMARK = 25  # we finally save maximum these many images per landmark
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import cv2

import random

import imageio

import imagehash

from PIL import Image

import seaborn as sns

from tqdm.auto import tqdm

import matplotlib.pyplot as plt
TRAIN_PATH = "../input/landmark-recognition-2020/train"



def id2filename(imageid):

    sid = str(imageid)

    return f"{TRAIN_PATH}/{sid[0]}/{sid[1]}/{sid[2]}/{sid}.jpg"





def plot_matches(match_dict, idx=-1):

    pair = match_dict[idx]

    id1, id2 = pair[0].split('_')

    img1 = imageio.imread(id2filename(id1))

    img2 = imageio.imread(id2filename(id2))

    

    plt.figure()

    plt.suptitle(f"#hash diff {pair[1]}")

    plt.subplot(121);plt.imshow(img1)

    plt.subplot(122);plt.imshow(img2)

    

    

def get_matches(landmark_id):

    mask = df['landmark_id'] == landmark_id

    image_names = df[mask]['id'].values.tolist()

    num_samples = min(len(image_names), NUM_SAMPLES_TO_COMPARE)

    image_names = random.sample(image_names, num_samples)

    img_hash = []

    for name in image_names:

        img_hash.append(imagehash.dhash(Image.open(id2filename(name))))

        

    match_pct = {}

    for i in range(len(img_hash)):

        base_name = image_names[i]

        for j in range(i+1, len(img_hash)):

            curr_name = image_names[j]

            match_pct[f"{base_name}_{curr_name}"] = img_hash[i] - img_hash[j]

    

    match_pct = sorted(match_pct.items(), key=lambda item: item[1])

    return match_pct
df = pd.read_csv('../input/landmark-recognition-2020/train.csv')

df
%%time



for landmark_id in [20409, 83144, 138982]:

    print(landmark_id)

    match_pct = get_matches(landmark_id)

    plt.figure(); plot_matches(match_pct, idx=0)

    plt.figure(); plot_matches(match_pct, idx=-1)
count_df = df.groupby('landmark_id').count().reset_index().rename(columns={'id': 'count'})

count_df = count_df.sort_values(by=['count'], ascending=False)

count_df
count_df['count'].hist(bins=100, figsize=(10, 4))
count_df['count'][100:].hist(bins=100, figsize=(10, 4))
count_mask = count_df['count'] >= NUM_SAMPLES_SAVE_PER_LANDMARK

count_mask.sum(), (~count_mask).sum()
from multiprocessing import Pool



def f(landmark_id):

    match_pct = get_matches(landmark_id)

    

    filenames = set()

    for idx in range(len(match_pct)):

        pair = match_pct[-idx]  # sorted in ascending order of hamming distance

        id1, id2 = pair[0].split('_')

        filenames.add((id1, landmark_id))

        filenames.add((id2, landmark_id))

        

        # ensuring an upper cap to the number

        if len(filenames) >= NUM_SAMPLES_SAVE_PER_LANDMARK:

            break

    return filenames

    



if __name__ == '__main__':

    with Pool(5) as p:

        values = count_df[count_mask]['landmark_id']

        l = list(tqdm(p.imap(f, values), total=len(values)))

    

samples_df = []

for filenames in l:

    samples_df += list(filenames)

samples_df = pd.DataFrame(samples_df, columns=['id', 'landmark_id'])
unsampled_landmarks = count_df[(~count_mask)]['landmark_id']  

unsamled_mask = df['landmark_id'].isin(unsampled_landmarks)

unsampled_landmarks_df = df[unsamled_mask]

unsampled_landmarks_df
final_df = pd.concat([samples_df, unsampled_landmarks_df])

final_df.to_csv(f'train_reduced_to_{NUM_SAMPLES_SAVE_PER_LANDMARK}_samples.csv')
final_count_df = final_df.groupby('landmark_id').count().reset_index().rename(columns={'id': 'count'})

final_count_df = final_count_df.sort_values(by=['count'], ascending=False)

final_count_df['count'].hist();
final_df['landmark_id'].unique().shape == df['landmark_id'].unique().shape
final_df.shape, df.shape