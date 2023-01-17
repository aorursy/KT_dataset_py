from PIL import Image

import os

import cv2

import numpy as np

import pandas as pd

import seaborn as sns



from sklearn.metrics import log_loss

from PIL import Image, ImageDraw

import torch

import torch.nn as nn

from torch import optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader, Dataset, Subset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib.pylab as plt



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip install /kaggle/input/face-recognition/dlib-19.19.0.tar.gz
!pip install /kaggle/input/face-recognition/face_recognition_models-0.3.0.tar.gz

!pip install /kaggle/input/face-recognition/face_recognition-1.3.0-py2.py3-none-any.whl

!pip install /kaggle/input/face-recognition/face_recognition-1.3.0/dist/face_recognition-1.3.0.tar
train_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').transpose()

train_metadata.head()
print("PyTorch version:", torch.__version__)

print("CUDA version:", torch.version.cuda)

print("cuDNN version:", torch.backends.cudnn.version())
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gpu
train_metadata.groupby('label')['label'].count().plot(figsize=(15, 5), kind='bar', title='Distribution of Labels in the Training Set')

plt.show()
train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'

fig, ax = plt.subplots(1,1, figsize=(15, 15))

train_video_files = [train_dir + x for x in os.listdir(train_dir)]

video_file = train_video_files[30]

cap = cv2.VideoCapture(video_file)

success, image = cap.read()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cap.release()   

ax.imshow(image)

ax.xaxis.set_visible(False)

ax.yaxis.set_visible(False)

plt.grid(False)
import face_recognition

face_recog = face_recognition.face_locations(image)

from PIL import Image

print("I found{} face(s) in the photograph".format(len(face_recog)))



for face_location in face_recog:

    top, right, bottom, left =  face_location

    print("face is located Top:{}, Left:{}, Bottom:{}, Right:{}".format(top, left, bottom, right))

    face_image = image[top:bottom, left:right]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    plt.grid(False)

    ax.xaxis.set_visible(False)

    ax.yaxis.set_visible(False)

    ax.imshow(face_image)
face_list =  face_recognition.face_landmarks(image)
from PIL import Image, ImageDraw

pil_image = Image.fromarray(image)

p = ImageDraw.Draw(pil_image)



for face_marks in face_list:

    for facial_data in face_marks.keys():

        print("point{}".format(facial_data))
for facial_data in face_marks.keys():

    p.line(face_marks[facial_data], width=4)



display(pil_image)
fig, axs = plt.subplots(19, 2, figsize=(15, 80))

axs = np.array(axs)

axs = axs.reshape(-1)

i = 0

for fn in train_metadata.index[:23]:

    label = train_metadata.loc[fn]['label']

    orig = train_metadata.loc[fn]['label']

    video_file = f'/kaggle/input/deepfake-detection-challenge/train_sample_videos/{fn}'

    ax = axs[i]

    cap = cv2.VideoCapture(video_file)

    success, image = cap.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(image)

    if len(face_locations) > 0:

        # Print first face

        face_location = face_locations[0]

        top, right, bottom, left = face_location

        face_image = image[top:bottom, left:right]

        ax.imshow(face_image)

        ax.grid(False)

        ax.title.set_text(f'{fn} - {label}')

        ax.xaxis.set_visible(False)

        ax.yaxis.set_visible(False)

        # Find landmarks

        face_landmarks_list = face_recognition.face_landmarks(face_image)

        face_landmarks = face_landmarks_list[0]

        pil_image = Image.fromarray(face_image)

        d = ImageDraw.Draw(pil_image)

        for facial_feature in face_landmarks.keys():

            d.line(face_landmarks[facial_feature], width=2)

        landmark_face_array = np.array(pil_image)

        ax2 = axs[i+1]

        ax2.imshow(landmark_face_array)

        ax2.grid(False)

        ax2.title.set_text(f'{fn} - {label}')

        ax2.xaxis.set_visible(False)

        ax2.yaxis.set_visible(False)

        i += 2

plt.grid(False)

plt.show()
sample_sub = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")

sample_sub['label'] = 0.5

sample_sub.loc[sample_sub['filename'] == 'aassnaulhq.mp4', 'label'] = 0 # Guess the true value

sample_sub.loc[sample_sub['filename'] == 'aayfryxljh.mp4', 'label'] = 0

sample_sub.to_csv('submission.csv', index=False)
sample_sub.head()
sample_sub.to_csv("submission.csv", index=False)