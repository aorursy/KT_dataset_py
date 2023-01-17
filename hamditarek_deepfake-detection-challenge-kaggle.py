# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from skimage.color import rgb2gray

import cv2

import matplotlib.pyplot as plt

%matplotlib inline

from scipy import ndimage

import os

import sys

import random

import math

import numpy as np

import skimage.io

import matplotlib

import matplotlib.pyplot as plt
train_sample_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T

train_sample_metadata.head()
train_sample_metadata.groupby('label')['label'].count().plot(figsize=(15, 5), kind='bar', title='Distribution of Labels in the Training Set')

plt.show()
from IPython.display import HTML

from base64 import b64encode

vid1 = open('/kaggle/input/deepfake-detection-challenge/test_videos/ytddugrwph.mp4','rb').read()

data_url = "data:video/mp4;base64," + b64encode(vid1).decode()

HTML("""

<video width=600 controls>

      <source src="%s" type="video/mp4">

</video>

""" % data_url)
vid3 = open('/kaggle/input/deepfake-detection-challenge/test_videos/acazlolrpz.mp4','rb').read()

data_url = "data:video/mp4;base64," + b64encode(vid3).decode()

HTML("""

<video width=600 controls>

      <source src="%s" type="video/mp4">

</video>

""" % data_url)
vid4 = open('/kaggle/input/deepfake-detection-challenge/test_videos/adohdulfwb.mp4','rb').read()

data_url = "data:video/mp4;base64," + b64encode(vid4).decode()

HTML("""

<video width=600 controls>

      <source src="%s" type="video/mp4">

</video>

""" % data_url)
import cv2



VIDEO_STREAM = "/kaggle/input/deepfake-detection-challenge/test_videos/ytddugrwph.mp4"

#VIDEO_STREAM_OUT = "/kaggle/input/deepfake-detection-challenge/test_videos/Result.mp4"



vidcap = cv2.VideoCapture(VIDEO_STREAM)

def getFrame(sec):

    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)

    hasFrames,image = vidcap.read()

    if hasFrames:

        cv2.imwrite("image"+str(count)+".jpg", image) # save frame as JPG file

        plt.imshow(image)

        



    

    return hasFrames

sec = 0

frameRate = 0.5 #//it will capture image in each 0.5 second

count=1

success = getFrame(sec)

while success:

    count = count + 1

    sec = sec + frameRate

    sec = round(sec, 2)

    success = getFrame(sec)
pic = plt.imread('image1.jpg')

print(pic.shape)

plt.imshow(pic)
pic = plt.imread('image2.jpg')

gray = rgb2gray(pic)

plt.imshow(gray, cmap='gray')
first_Video = "/kaggle/input/deepfake-detection-challenge/test_videos/ytddugrwph.mp4"
count = 0

cap = cv2.VideoCapture(first_Video)

ret,frame = cap.read()



while count < 3:

    cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))   

    ret,frame = cap.read()

    if count == 0:

        image0 = frame

    elif count == 1:

        image1 = frame

    elif count == 2:

        image2 = frame

    

    #cv2.imwrite( filepath+ "\frame%d.jpg" % count, image)     # Next I will save frame as JPEG

    count = count + 1
def display(img):

    

    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(111)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax.imshow(img)
display(image0)  # frame 1
display(image1)  # frame 2
display(image2)  # frame 3
import cv2 as cv

import os

import matplotlib.pylab as plt

train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'

fig, ax = plt.subplots(1,1, figsize=(15, 15))

train_video_files = [train_dir + x for x in os.listdir(train_dir)]

# video_file = train_video_files[30]

video_file = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/afoovlsmtx.mp4'

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
from PIL import Image, ImageDraw



fig, axs = plt.subplots(19, 2, figsize=(15, 80))

axs = np.array(axs)

axs = axs.reshape(-1)

i = 0

for fn in train_sample_metadata.index[:23]:

    label = train_sample_metadata.loc[fn]['label']

    orig = train_sample_metadata.loc[fn]['label']

    video_file = f'/kaggle/input/deepfake-detection-challenge/train_sample_videos/{fn}'

    ax = axs[i]

    cap = cv.VideoCapture(video_file)

    success, image = cap.read()

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

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
fig, axs = plt.subplots(19, 2, figsize=(10, 80))

axs = np.array(axs)

axs = axs.reshape(-1)

i = 0

pad = 60

for fn in train_sample_metadata.index[23:44]:

    label = train_sample_metadata.loc[fn]['label']

    orig = train_sample_metadata.loc[fn]['label']

    video_file = f'/kaggle/input/deepfake-detection-challenge/train_sample_videos/{fn}'

    ax = axs[i]

    cap = cv.VideoCapture(video_file)

    success, image = cap.read()

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(image)

    if len(face_locations) > 0:

        # Print first face

        face_location = face_locations[0]

        top, right, bottom, left = face_location

        face_image = image[top-pad:bottom+pad, left-pad:right+pad]

        ax.imshow(face_image)

        ax.grid(False)

        ax.title.set_text(f'{fn} - {label}')

        ax.xaxis.set_visible(False)

        ax.yaxis.set_visible(False)

        # Find landmarks

        face_landmarks_list = face_recognition.face_landmarks(face_image)

        try:

            face_landmarks = face_landmarks_list[0]

            pil_image = Image.fromarray(face_image)

            d = ImageDraw.Draw(pil_image)

            for facial_feature in face_landmarks.keys():

                d.line(face_landmarks[facial_feature], width=2, fill='yellow')

            landmark_face_array = np.array(pil_image)

            ax2 = axs[i+1]

            ax2.imshow(landmark_face_array)

            ax2.grid(False)

            ax2.title.set_text(f'{fn} - {label}')

            ax2.xaxis.set_visible(False)

            ax2.yaxis.set_visible(False)

            i += 2

        except:

            pass

plt.grid(False)

plt.tight_layout()

plt.show()
%%capture

# Install facenet-pytorch

!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-1.0.1-py3-none-any.whl



# Copy model checkpoints to torch cache so they are loaded automatically by the package

!mkdir -p /tmp/.cache/torch/checkpoints/

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth /tmp/.cache/torch/checkpoints/vggface2_DG3kwML46X.pt

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth /tmp/.cache/torch/checkpoints/vggface2_G5aNV2VSMn.pt
import os

import glob

import torch

import cv2

from PIL import Image

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



# See github.com/timesler/facenet-pytorch:

from facenet_pytorch import MTCNN, InceptionResnetV1



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f'Running on device: {device}')
# Load face detector

mtcnn = MTCNN(device=device).eval()



# Load facial recognition model

resnet = InceptionResnetV1(pretrained='vggface2', num_classes=2, device=device).eval()
# Get all test videos

filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')



# Number of frames to sample (evenly spaced) from each video

n_frames = 10



X = []

with torch.no_grad():

    for i, filename in enumerate(filenames):

        print(f'Processing {i+1:5n} of {len(filenames):5n} videos\r', end='')

        

        try:

            # Create video reader and find length

            v_cap = cv2.VideoCapture(filename)

            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            

            # Pick 'n_frames' evenly spaced frames to sample

            sample = np.linspace(0, v_len - 1, n_frames).round().astype(int)

            imgs = []

            for j in range(v_len):

                success, vframe = v_cap.read()

                vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)

                if j in sample:

                    imgs.append(Image.fromarray(vframe))

            v_cap.release()

            

            # Pass image batch to MTCNN as a list of PIL images

            faces = mtcnn(imgs)

            

            # Filter out frames without faces

            faces = [f for f in faces if f is not None]

            faces = torch.stack(faces).to(device)

            

            # Generate facial feature vectors using a pretrained model

            embeddings = resnet(faces)

            

            # Calculate centroid for video and distance of each face's feature vector from centroid

            centroid = embeddings.mean(dim=0)

            X.append((embeddings - centroid).norm(dim=1).cpu().numpy())

        except KeyboardInterrupt:

            raise Exception("Stopped.")

        except:

            X.append(None)
bias = -0.4

weight = 0.068235746



submission = []

for filename, x_i in zip(filenames, X):

    if x_i is not None and len(x_i) == 10:

        prob = 1 / (1 + np.exp(-(bias + (weight * x_i).sum())))

    else:

        prob = 0.6

    submission.append([os.path.basename(filename), prob])
submission = pd.DataFrame(submission, columns=['filename', 'label'])

submission.sort_values('filename').to_csv('submission.csv', index=False)
plt.hist(submission.label, 20)

plt.show()

submission