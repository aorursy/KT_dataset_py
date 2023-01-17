# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import json



meta = pd.DataFrame(json.load(open('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json'))).T

print(meta.head())





# Any results you write to the current directory are saved as output.
meta['label'].value_counts()
meta[meta['label']=='REAL'].head()
# read a random real video in memory and analyse

import random

try:

    import imageio

    import pylab

except Exception as e:

    !pip install imageio

    !pip install pylab

    import pylab

    import imageio

!pip install imageio-ffmpeg

real_vids = meta[(meta['label'] == 'REAL') & (meta['split'] == 'train')]

rand_real_vid = real_vids.index[random.randint(0,len(real_vids))] 

filename = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'+rand_real_vid

print(filename)

vid = imageio.get_reader(filename,  'ffmpeg')

type(vid)
vid.count_frames()
image = vid.get_data(1)

pylab.imshow(image)
#Find number of frames in each video

import tqdm

frames_per_video = list()

for file in tqdm.tqdm(meta.index):

    filename = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'+file

    vid = imageio.get_reader(filename,  'ffmpeg')

    frames_per_video.append(vid.count_frames())
import matplotlib.pyplot as plt

plt.plot(frames_per_video)

plt.show()
# read meta data info about each video

import tqdm

meta_per_video = list()

for file in tqdm.tqdm(meta.index):

    filename = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'+file

    vid = imageio.get_reader(filename,  'ffmpeg')

    vid_meta = vid._meta

    # append other info from meta file

    vid_meta['num_frames'] = vid.count_frames()

    vid_meta['filename'] = file

    vid_meta['label'] = meta.loc[file]['label']

    vid_meta['split'] = meta.loc[file]['split']

    vid_meta['original'] = meta.loc[file]['original']

    meta_per_video.append(vid_meta)

#convert list of dict to pandas dataframe for easy analysis

meta_videos = pd.DataFrame(meta_per_video)
meta_videos.head()
list(filter(lambda x:x.split('.')[1]!='mp4',os.listdir('../input/deepfake-detection-challenge/test_videos')))


meta_per_video = list()

for file in tqdm.tqdm(os.listdir('../input/deepfake-detection-challenge/test_videos')):

    filename = '/kaggle/input/deepfake-detection-challenge/test_videos/'+file

    vid = imageio.get_reader(filename,  'ffmpeg')

    vid_meta = vid._meta

    # append other info from meta file

    vid_meta['num_frames'] = vid.count_frames()

    vid_meta['filename'] = file

    meta_per_video.append(vid_meta)

#convert list of dict to pandas dataframe for easy analysis

meta_test_videos = pd.DataFrame(meta_per_video)
meta_test_videos.head()
plt.plot(meta_test_videos['num_frames'])

plt.show()
#import the libraries 

import PIL.Image

import PIL.ImageDraw

try:

    import face_recognition

except:

    !pip install face_recognition

    import face_recognition

    

# load a video

vid = imageio.get_reader('/kaggle/input/deepfake-detection-challenge/train_sample_videos/aagfhgtpmv.mp4',  'ffmpeg')



# get a random frame of video

image = vid.get_data(random.randint(0,vid.count_frames()))



# Load the jpg file into a NumPy array

#image = face_recognition.load_image_file(image)



# Find all the faces in the image

face_locations = face_recognition.face_locations(image)



number_of_faces = len(face_locations)

print("I found {} face(s) in this photograph.".format(number_of_faces))



# Load the image into a Python Image Library object so that we can draw on top of it and display it

pil_image = PIL.Image.fromarray(image)



for face_location in face_locations:



    # Print the location of each face in this image. Each face is a list of co-ordinates in (top, right, bottom, left) order.

    top, right, bottom, left = face_location

    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # Let's draw a box around the face

    draw = PIL.ImageDraw.Draw(pil_image)

    draw.rectangle([left, top, right, bottom], outline="red")



# Display the image on screen

pil_image.show()
