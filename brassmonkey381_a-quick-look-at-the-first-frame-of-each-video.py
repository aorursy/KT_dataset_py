# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print('directory contents:', ', '.join(os.listdir('/kaggle/input/deepfake-detection-challenge')))



print(

    'num train videos:', len(os.listdir('/kaggle/input/deepfake-detection-challenge/train_sample_videos/')) - 1,

    '\nnum test videos: ',  len(os.listdir('/kaggle/input/deepfake-detection-challenge/test_videos/'))

)
import cv2 as cv

from matplotlib import pyplot as plt

from tqdm import tqdm
train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'

train_video_files = [train_dir + x for x in os.listdir(train_dir) if x.endswith('.mp4')]

test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/'

test_video_files = [test_dir + x for x in os.listdir(test_dir)]
train_metadata = pd.read_json('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json')

train_metadata = train_metadata.T

train_metadata.head()
train_metadata['label'].value_counts(normalize=True)
def show_first_frame(video_files, num_to_show=25):

    root = int(num_to_show**.5)

    fig, axes = plt.subplots(root,root, figsize=(root*5,root*5))

    for i, video_file in tqdm(enumerate(video_files[:num_to_show]), total=num_to_show):

        cap = cv.VideoCapture(video_file)

        success, image = cap.read()

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        cap.release()   

        

        axes[i//root, i%root].imshow(image)

        fname = video_file.split('/')[-1]        

        try:

            label = train_metadata.loc[fname, 'label']

            axes[i//root, i%root].title.set_text(f"{fname}: {label}")

        except:

            axes[i//root, i%root].title.set_text(f"{fname}")
show_first_frame(train_video_files, num_to_show=25)
show_first_frame(test_video_files, num_to_show=25)
fig, ax = plt.subplots(1,1, figsize=(12,12))

cap = cv.VideoCapture(test_dir + 'ahjnxtiamx.mp4')

cap.set(1,2)

success, image = cap.read()

image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

cap.release()   



ax.imshow(image)

fname = 'ahjnxtiamx.mp4'

ax.title.set_text(f"{fname}")