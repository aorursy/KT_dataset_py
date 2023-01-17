import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot  as plt # data visualization

import os



# Dataset parameters:

INPUT_DIR = os.path.join('..', 'input')



DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-recognition-2020')

TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')

TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train')

TRAIN_LABELMAP_PATH = os.path.join(DATASET_DIR, 'train.csv')

train = pd.read_csv(f'{DATASET_DIR}/train.csv')
print("Shape of train_data :", train.shape)

print("Number of unique landmarks :", train["landmark_id"].nunique())
train.head()
idx = train.id[1]

idx
def get_image_full_path(idx):

    return os.path.join(TRAIN_IMAGE_DIR,  f'{idx[0]}/{idx[1]}/{idx[2]}/{idx}.jpg')
from PIL import Image, ImageDraw



image = Image.open(get_image_full_path(idx))

plt.imshow(image) 

image.close()       

plt.axis("off")



plt.show() 

example = train[train["landmark_id"]==1]

for idx in example["id"]:

    image = Image.open(get_image_full_path(idx))

    plt.imshow(image) 

    image.close()       

    plt.axis("off")

    plt.show() 

!pip install basic_image_eda

from basic_image_eda import BasicImageEDA
data_dir = "../input/landmark-recognition-2020/train/0"

extensions = ['png', 'jpg', 'jpeg']

threads = 0

dimension_plot = True

channel_hist = True

nonzero = False

hw_division_factor = 1.0



BasicImageEDA.explore(data_dir, extensions, threads, dimension_plot, channel_hist, nonzero, hw_division_factor)
import seaborn as sns 



ad = sns.distplot(train['landmark_id'].value_counts()[-75000:])

ad.set(xlabel='Landmark Counts', ylabel='Probability Density', title='Distribution of less common landmarks')

plt.show()