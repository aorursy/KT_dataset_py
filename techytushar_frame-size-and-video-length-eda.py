import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm
path = '/kaggle/input/deepfake-detection-challenge'

# reading file names

train_files = os.listdir(os.path.join(path, 'train_sample_videos'))

train_files.remove('metadata.json')

test_files = os.listdir(os.path.join(path, 'test_videos'))



print(f'Number of Train files: {len(train_files)}\nNumber of Test files: {len(test_files)}')
# reading the labels json file

labels_df = pd.read_json(os.path.join(path, 'train_sample_videos/metadata.json'))

labels_df = labels_df.T

print(labels_df.shape)

labels_df.head()
# number of real and fake samples

labels_df['label'].value_counts(normalize=True)*100
# gets the frame size for a video

def get_frame_size(file):

    cap = cv2.VideoCapture(file)

    ret, frame = cap.read()

    #plt.imshow(frame)

    shape = frame.shape

    cap.release()

    return shape



# gets the fps and duration of video

def get_video_length(file):

    cap = cv2.VideoCapture(file)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count/fps

    cap.release()

    return round(fps), round(duration)



# extract metadata of all files and return in a dataframe

def extract_metadata(files, path):

    frame_size_list = []

    fps_list = []

    duration_list = []

    for i in tqdm(files):

        shape = get_frame_size(os.path.join(path,f'{i}'))

        fps, duration = get_video_length(os.path.join(path,f'{i}'))

        frame_size_list.append(shape)

        fps_list.append(fps)

        duration_list.append(duration)



    meta_df = pd.DataFrame(data={'frame_shape':frame_size_list, 'fps':fps_list, 'duration':duration_list}, index=files)

    return meta_df



print(get_frame_size(os.path.join(path, 'train_sample_videos/aagfhgtpmv.mp4')))

print(get_video_length(os.path.join(path, 'train_sample_videos/aagfhgtpmv.mp4')))
# getting metadata for train files

train_meta = extract_metadata(train_files, os.path.join(path, 'train_sample_videos'))

train_meta.head()
train_meta.frame_shape.value_counts()
train_meta.fps.value_counts()
print('Duration in seconds')

print(train_meta.duration.value_counts())
# getting metadata for test files

test_meta = extract_metadata(test_files, os.path.join(path, 'test_videos'))

test_meta.head()
test_meta.frame_shape.value_counts()
test_meta.fps.value_counts()
test_meta.duration.value_counts()
submission = pd.read_csv(f"{path}/sample_submission.csv")

submission.head()
submission['label'] = 0.7

submission.to_csv('submission.csv', index=False)