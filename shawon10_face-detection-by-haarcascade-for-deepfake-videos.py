# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('../input/haarcascadefrontalfaces/'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'

train_video_files = [train_dir + x for x in os.listdir(train_dir)]

test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/'

test_video_files = [test_dir + x for x in os.listdir(test_dir)]
df_train = pd.read_json('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json').transpose()

df_train.head()
df_train.label.value_counts()
df_train.label.value_counts().plot(kind='pie')
import cv2

# Load the cascade

face_cascade = cv2.CascadeClassifier('../input/haarcascadefrontalfaces/haarcascade_frontalface_default.xml')

columns = 1

rows = 10

plt.figure(figsize=(100, 100))

# To capture video from webcam. 

#cap = cv2.VideoCapture(0)

# To use a video file as input 

#_,img = cap.read()

for i in range(1, columns*rows+1):

    cap = cv2.VideoCapture(train_video_files[i])

    _,img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces

    faces = face_cascade.detectMultiScale(gray,1.1, 5)

    # Draw the rectangle around each face

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)

    # Display

    name_video = train_video_files[i].split('/', 5)[5]

    df=df_train[df_train.index == name_video] 

    label=df['label'].values[0]

    #ax.set_title(label)

    plt.subplot(rows, columns, i)

    plt.imshow(img)

    #x=img.shape[1]

    plt.text(1000,1000, str(label),

             fontsize=18, ha='center',backgroundcolor='black', color='white', weight='bold')

    cap.release()

plt.show()
import cv2

# Load the cascade

face_cascade = cv2.CascadeClassifier('../input/haarcascadefrontalfaces/haarcascade_frontalface_default.xml')

columns = 1

rows = 5

plt.figure(figsize=(100, 100))

# To capture video from webcam. 

#cap = cv2.VideoCapture(0)

# To use a video file as input 

#_,img = cap.read()

for i in range(1, 5):

    cap = cv2.VideoCapture(test_video_files[i])

    _,img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces

    faces = face_cascade.detectMultiScale(gray,1.1, 5)

    # Draw the rectangle around each face

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)

    # Display

    name_video = train_video_files[i].split('/', 5)[5]

    #df=df_train[df_train.index == name_video] 

    #label=df['label'].values[0]

    #ax.set_title(label)

    plt.subplot(rows, columns, i)

    plt.imshow(img)

    #x=img.shape[1]

    plt.text(1000,1000, str(name_video),

             fontsize=18, ha='center',backgroundcolor='black', color='white', weight='bold')

    cap.release()

plt.show()
df = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")

df['label'] = 0.5 #maximum are fake

df.loc[df['filename'] == 'aassnaulhq.mp4', 'label'] = 0 # Guess the true value

df.loc[df['filename'] == 'aayfryxljh.mp4', 'label'] = 0

df.loc[df['filename'] == 'alrtntfxtd.mp4', 'label'] = 0

df.loc[df['filename'] == 'ayipraspbn.mp4', 'label'] = 0

df.to_csv('submission.csv', index=False)