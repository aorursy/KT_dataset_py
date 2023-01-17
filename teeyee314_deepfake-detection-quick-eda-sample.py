import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import torch

import keras

import glob

import cv2

from albumentations import *

from tqdm import tqdm_notebook as tqdm

import gc



import warnings

warnings.filterwarnings('ignore')

PATH = '../input/deepfake-detection-challenge/'

print(os.listdir(PATH))
TRAIN_PATH = 'train_sample_videos/*.mp4'

TEST_PATH = 'test_videos/*.mp4'

train_img = glob.glob(os.path.join(PATH, TRAIN_PATH))

test_img = glob.glob(os.path.join(PATH, TEST_PATH))
def gather_info(train_img):

    train_img = os.path.join(PATH, f"train_sample_videos/{train_img}")



    cap = cv2.VideoCapture(train_img)



    success, image = cap.read()

    count = 0

    first_trn_image = None



    while success:

        try:

            success, image = cap.read()

            if count == 0:

                first_trn_image = image

            x, y, z = image.shape



        except:

            break

        count += 1

        

    cap.release()

    cv2.destroyAllWindows()

    

    return [x,y,z,count], first_trn_image
meta = glob.glob(os.path.join(PATH, 'train_sample_videos/*.json'))

label_df = pd.read_json(meta[0]).transpose().reset_index()

idx = label_df['index']

image_meta = []

first_trn_images = []
label_df.head()
# gather training image dimensions

for i in tqdm(range(len(idx))):

    test, first = gather_info(idx.values[i])

    image_meta.append(test)

    first_trn_images.append(first)
# Display the first image(frame) of each video(.mp4) in sample train

for img, lbl in zip(first_trn_images[:100], label_df['label'].values[:100]):

    # cv2's native channel is BGR, we need to convert to RGB

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.title(f'{lbl}')

    plt.imshow(img)

    plt.show()

    

gc.collect()
image_meta = np.array(image_meta, dtype='int16')

label_df['height'] = image_meta[:,0]

label_df['width'] = image_meta[:,1]

label_df['channel'] = image_meta[:,2]

label_df['frames'] = image_meta[:,3]
label_df.head()
label_df['height'].value_counts()
label_df['width'].value_counts()
label_df['channel'].value_counts()
label_df['frames'].value_counts()
# Sample Train Target Distribution

plt.title('Count of each ship type')

sns.countplot(y=label_df['label'].values)

plt.show()
def gather_test(test_img, i):

    cap = cv2.VideoCapture(test_img[i])



    success, image = cap.read()

    count = 0

    first_tst_image = None



    while success:

        try:

            success, image = cap.read()

            if count == 0:

                first_tst_image = image

            x, y, z = image.shape



        except:

            break

        count += 1

        

    cap.release()

    cv2.destroyAllWindows()

    

    return [x,y,z,count], first_tst_image
test_image_meta = []

first_tst_images = []
# gather test image dimensions

for i in tqdm(range(len(test_img))):

    test, first = gather_test(test_img, i)

    test_image_meta.append(test)

    first_tst_images.append(first)
# Display the first image(frame) of each video(.mp4) in test

for img in first_tst_images[:100]:

    # cv2's native channel is BGR, we need to convert to RGB

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)

    plt.show()

    

gc.collect()
test_image_meta = np.array(test_image_meta)

test_meta = pd.DataFrame({'height':test_image_meta[:,0],

                          'width':test_image_meta[:,1],

                          'channel':test_image_meta[:,2],

                          'frames':test_image_meta[:,3]})

test_meta.head()
test_meta['height'].value_counts()
test_meta['width'].value_counts()
test_meta['channel'].value_counts()
test_meta['frames'].value_counts()