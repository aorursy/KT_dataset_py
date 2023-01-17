import os

import random



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2
df = pd.read_csv('../input/landmark-retrieval-2020/train.csv', index_col=0)
def create_full_path(name):

    return os.path.join(

        '../input/landmark-retrieval-2020/train/',

        name[0],

        name[1],

        name[2],

        f'{name}.jpg'

    )
def vis(_id):

    arr = df[df['landmark_id'] == _id].index.tolist()



    plt.figure(figsize=(16, 16))

    for i, name in enumerate(arr):

        img = cv2.imread(create_full_path(name))

        plt.subplot(4, 3, i + 1)

        plt.imshow(img)

        plt.xticks([])

        plt.yticks([])

        if i >= 11:

            break

    plt.suptitle(_id)

    plt.show()
vis(random.choice(df['landmark_id'].unique()))
vis(random.choice(df['landmark_id'].unique()))
vis(random.choice(df['landmark_id'].unique()))
vis(random.choice(df['landmark_id'].unique()))
vis(random.choice(df['landmark_id'].unique()))
vis(random.choice(df['landmark_id'].unique()))
vis(random.choice(df['landmark_id'].unique()))
vis(random.choice(df['landmark_id'].unique()))
vis(random.choice(df['landmark_id'].unique()))