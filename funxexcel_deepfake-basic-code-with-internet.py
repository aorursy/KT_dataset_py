import numpy as np

import pandas as pd

import matplotlib.pylab as plt

import cv2

plt.style.use('ggplot')

from IPython.display import Video

from IPython.display import HTML
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# SKLearn Implemention

from sklearn.metrics import log_loss

log_loss(["REAL", "FAKE", "FAKE", "REAL"],

         [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
!ls -GFlash ../input/deepfake-detection-challenge
!du -sh ../input/deepfake-detection-challenge/
train_sample_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T

train_sample_metadata.head()
train_sample_metadata.groupby('label')['label'].count().plot(figsize=(15, 5), kind='bar', title='Distribution of Labels in the Training Set')

plt.show()
import cv2 as cv

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
!pip install face_recognition
import face_recognition

face_locations = face_recognition.face_locations(image)



# https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py

from PIL import Image



print("I found {} face(s) in this photograph.".format(len(face_locations)))



for face_location in face_locations:



    # Print the location of each face in this image

    top, right, bottom, left = face_location

    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))



    # You can access the actual face itself like this:

    face_image = image[top:bottom, left:right]

    fig, ax = plt.subplots(1,1, figsize=(5, 5))

    plt.grid(False)

    ax.xaxis.set_visible(False)

    ax.yaxis.set_visible(False)

    ax.imshow(face_image)
ss = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")

ss['label'] = 0.5

ss.loc[ss['filename'] == 'aassnaulhq.mp4', 'label'] = 0 # Guess the true value

ss.loc[ss['filename'] == 'aayfryxljh.mp4', 'label'] = 0

ss.to_csv('submission.csv', index=False)
ss.head()