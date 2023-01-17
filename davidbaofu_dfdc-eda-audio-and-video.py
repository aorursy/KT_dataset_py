import numpy as np

import pandas as pd

import os

import random

import subprocess

from pathlib import Path

import IPython



import seaborn as sns

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm

tqdm.pandas()



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
metadata = pd.read_csv('../input/train-set-metadata-for-dfdc/metadata', low_memory=False)

metadata.head()
def audio_label(row): 

    if row['label'] == 'REAL':

        return 'REAL'

    if row['wav.hash'] != row['wav.hash.orig'] and row['audio.@codec_time_base'] != '1/16000':

        return 'FAKE'

    return 'REAL'



def video_label(row):

    if row['label'] == 'REAL':

        return 'REAL'

    if row['pxl.hash'] != row['pxl.hash.orig']:

        return 'FAKE'

    return 'REAL'
metadata["video_label"] = metadata.progress_apply(video_label, axis=1)

metadata["audio_label"] = metadata.progress_apply(audio_label, axis=1)
clean_labels = metadata[["filename", "video_label", "audio_label"]]
sns.set_style('darkgrid')



plt.figure(figsize=(12,5))

plt.title('Label Distribution')



plt.subplot(1, 3, 1)

ax1 = sns.countplot(metadata["video_label"], order=["REAL", "FAKE"])

plt.subplot(1, 3, 2)

ax2 = sns.countplot(metadata["audio_label"], order=["REAL", "FAKE"])



union_label = metadata["video_label"].str.cat(metadata["audio_label"], sep="_")



plt.subplot(1, 3, 3)

ax3 = sns.countplot(union_label)



ax1.set_ylim(0, 120000)

ax2.set_ylim(0, 120000)

ax3.set_ylim(0, 120000)



plt.show()
print(f"Number of both FAKE video and FAKE audio: {len(union_label[union_label == 'FAKE_FAKE'])}")

print(f"Number of only FAKE audio: {len(metadata[metadata['audio_label']=='FAKE'])}")
num_audio_fakes = metadata["audio_label"].value_counts()["FAKE"]

print(f"We only have {num_audio_fakes} fake audio samples. It is undersampled in comparison to other labels.")
path = "../input/deepfake-detection-challenge/train_sample_videos/"

videos = [os.path.join(path, video) for video in os.listdir(path)]
def get_video_label(path, metadata):

    filename = os.path.basename(path)

    data = metadata[metadata["filename"] == filename]

    return data["video_label"]



def get_audio_label(path, metadata):

    filename = os.path.basename(path)

    data = metadata[metadata["filename"] == filename]

    return data["audio_label"]
! tar xvf ../input/ffmpeg-static-build/ffmpeg-git-amd64-static.tar.xz
def create_audio(file, save_path):

    command = f"../working/ffmpeg-git-20191209-amd64-static/ffmpeg -i {file} -ab 192000 -ac 2 -ar 44100 -vn {save_path}"

    subprocess.call(command, shell=True)

    

output_format = "mp3"

output_dir = Path(f"mp3_files")

Path(output_dir).mkdir(exist_ok=True, parents=True)
def get_random_frame(path):

    cap = cv2.VideoCapture(path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, random.uniform(0, 1))

    _, img = cap.read()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img



def visualize_sample(sample):

    video_label = get_video_label(sample, clean_labels)

    audio_label = get_audio_label(sample, clean_labels)



    # Read random image

    img = get_random_frame(sample)

    plt.imshow(img)

    plt.title(f"Video: {video_label.item()}, Audio: {audio_label.item()}") # .item() works as of now 16/03/2020, but will be removed in the future

    plt.show()
sample = videos[143]



# Visualize random frame

visualize_sample(sample)



# Read audio

audio_file = f"{output_dir/sample[-14:-4]}.{output_format}"

create_audio(sample, audio_file)

IPython.display.Audio(audio_file)
sample = videos[203]



# Visualize random frame

visualize_sample(sample)



# Read audio

audio_file = f"{output_dir/sample[-14:-4]}.{output_format}"

create_audio(sample, audio_file)

IPython.display.Audio(audio_file)