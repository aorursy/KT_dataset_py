import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import file utilities

import os

import glob



# import charting

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, ArtistAnimation 

%matplotlib inline



from IPython.display import HTML



# import computer vision

import cv2

from skimage.measure import compare_ssim
df_test = '../input/deepfake-detection-challenge/test_videos/'

df_train = '../input/deepfake-detection-challenge/train_sample_videos/'

df_meta = '../input/deepfake-detection-challenge/train_sample_videos/metadata.json'
DATA_FOLDER = '../input/deepfake-detection-challenge'

TRAIN_SAMPLE_FOLDER = 'train_sample_videos'

TEST_FOLDER = 'test_videos'



print(f"Train samples: {len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")

print(f"Test samples: {len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")
df_meta = pd.read_json(df_meta).transpose()

df_meta.head()
df_meta['label'].value_counts(normalize=True)

train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))

ext_dict = []

for file in train_list:

    file_ext = file.split('.')[1]

    if (file_ext not in ext_dict):

        ext_dict.append(file_ext)

print(f"Extensions: {ext_dict}")
meta = np.array(list(df_meta.index))

storage = np.array([file for file in train_list if  file.endswith('mp4')])

print(f"Metadata: {meta.shape[0]}, Folder: {storage.shape[0]}")

print(f"Files in metadata and not in folder: {np.setdiff1d(meta,storage,assume_unique=False).shape[0]}")

print(f"Files in folder and not in metadata: {np.setdiff1d(storage,meta,assume_unique=False).shape[0]}")
import cv2 as cv

from tqdm import tqdm

import os

import matplotlib.pylab as plt

train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'

fig, ax = plt.subplots(1,1, figsize=(15, 15))

train_video_files = [train_dir + x for x in os.listdir(train_dir)]

# video_file = train_video_files[30]

video_file = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/akxoopqjqz.mp4'

cap = cv.VideoCapture(video_file)

success, image = cap.read()

image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

cap.release()   

ax.imshow(image)

ax.xaxis.set_visible(False)

ax.yaxis.set_visible(False)

ax.title.set_text(f"FRAME 0: {video_file.split('/')[-1]}")

plt.grid(False)
video_file = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/akxoopqjqz.mp4'



cap = cv2.VideoCapture(video_file)



frames = []

while(cap.isOpened()):

    ret, frame = cap.read()

    if ret==True:

        frames.append(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

    else:

        break

cap.release()



print('The number of frames saved: ', len(frames))
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
test_video_files = [df_test + x for x in os.listdir(df_test)]

show_first_frame(test_video_files, num_to_show=25)
fig, ax = plt.subplots(1,1, figsize=(12,12))

cap = cv.VideoCapture(df_test + 'ahjnxtiamx.mp4')

cap.set(1,2)

success, image = cap.read()

image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

cap.release()   



ax.imshow(image)

fname = 'ahjnxtiamx.mp4'

ax.title.set_text(f"{fname}")