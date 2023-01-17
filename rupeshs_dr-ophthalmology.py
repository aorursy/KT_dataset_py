import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

import tqdm

from PIL import Image

%matplotlib inline



train_df = pd.read_csv('../input/train.csv')
fig = plt.figure(figsize=(32, 32))

num_samples=4



for class_id in sorted(train_df['diagnosis'].unique()):

    for i, (idx, row) in enumerate(train_df.loc[train_df['diagnosis'] == class_id].sample(num_samples).iterrows()):

        ax = fig.add_subplot(5, num_samples, class_id *num_samples + i + 1, xticks=[], yticks=[])

        im = Image.open(f"../input/train_images/{row['id_code']}.png")

        plt.imshow(im)

        ax.set_title(f'Label: {class_id}')