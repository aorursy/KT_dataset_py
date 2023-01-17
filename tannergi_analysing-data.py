import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.patches as patches

from PIL import Image

import random
df = pd.read_csv('../input/microcontroller-detection/Microcontroller Detection/train_labels.csv')

df.head()
df['class'].value_counts().plot(kind="bar")
occure_more_than_once = df['filename'].value_counts()[df['filename'].value_counts()>2]

occure_more_than_once
fig = plt.figure(figsize=(20, 15))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

train_images_path = '../input/microcontroller-detection/Microcontroller Detection/train/'

for i in range(1, 8):

    ax = fig.add_subplot(3, 3, i)

    ax.imshow(Image.open(train_images_path + occure_more_than_once.index.values[i-1]))
fig, ax = plt.subplots(2, 3, figsize=(20, 15))



for i, r in enumerate(random.sample(range(len(df)), 6)):

    row = df.iloc[r]

    im = np.array(Image.open(train_images_path + row['filename']), dtype=np.uint8)



    # Display the image

    ax[int(i/3), i%3].imshow(im)

    

    for _, instance in df[(df['filename']==row['filename'])].iterrows():

        # Create a Rectangle patch

        rect = patches.Rectangle((instance['xmin'], instance['ymin']), instance['xmax'] - instance['xmin'], 

                                  instance['ymax'] - instance['ymin'], linewidth=3, edgecolor='r', facecolor='none')



        # Add the patch to the Axes

        ax[int(i/3), i%3].add_patch(rect)