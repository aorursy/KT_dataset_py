# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import os

from tqdm import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



print(os.getcwd())

for dirname, dirs, _ in os.walk('../input'):

    for d in dirs:

        print(os.path.join(dirname, d))



# Any results you write to the current directory are saved as output.
def video_prop(reader):

    w = 0

    h = 0



    success, image = reader.read()

    h = image.shape[0]

    w = image.shape[1]

    nFrames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    return h, w, nFrames

def video_size_counter(path):

    video_sizes = dict()

    for dirname, _, filenames in os.walk(path):

        for filename in filenames:

            if filename.endswith('.mp4'):

                video_filename = os.path.join(dirname, filename)

                reader = cv2.VideoCapture(video_filename)

                h, w, nFrames = video_prop(reader)

                if (h, w, nFrames) in video_sizes.keys():

                    video_sizes[(h, w, nFrames)] += 1

                else:

                    video_sizes[(h, w, nFrames)] = 1

    return video_sizes
video_sizes_train = video_size_counter('/kaggle/input/deepfake-detection-challenge/train_sample_videos')

video_sizes_test = video_size_counter('/kaggle/input/deepfake-detection-challenge/test_videos')
print(video_sizes_train)

print(video_sizes_test)

sizes1 = set([k for k in video_sizes_train.keys()])

sizes2 = set([k for k in video_sizes_test.keys()])

sizes = sizes1.union(sizes2)

sizes_str = [str(s) for s in sizes]

y_pos = [3*i for i in range(len(sizes_str))]



n_accurance = []

for s in sizes:

    if s in video_sizes_train.keys():

        n_accurance.append(video_sizes_train[s])

    else:

        n_accurance.append(0)

fig = plt.figure(1)

plt.bar(y_pos, n_accurance, width=1)

plt.xticks(y_pos, sizes_str)

plt.title('Video sizes distribution over the sampled training data')

plt.show()



n_accurance = []

for s in sizes:

    if s in video_sizes_test.keys():

        n_accurance.append(video_sizes_test[s])

    else:

        n_accurance.append(0)

fig = plt.figure(2)

plt.bar(y_pos, n_accurance, width=1)

plt.xticks(y_pos, sizes_str)

plt.title('Video sizes distribution over the sampled test data')

plt.show()
train_sample_metadata = pd.read_json('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json').T

train_sample_metadata.head(20)
labels_count = train_sample_metadata.groupby('label')['label'].count()

print(labels_count)

labels_count.plot(kind='bar', title='Distribution of Labels in the Training Set')

plt.show()
class ObjectDetector:

    # Class for Object Detection

    def __init__(self,object_cascade_path):

        '''

        param: object_cascade_path - path for the *.xml defining the parameters for {face, eye, smile, profile}

        detection algorithm

        source of the haarcascade resource is: https://github.com/opencv/opencv/tree/master/data/haarcascades

        '''

        self.objectCascade = cv2.CascadeClassifier(object_cascade_path)



    def detect(self, image, scale_factor=1.3, min_neighbors=5, min_size=(20,20)):

        '''

        Function return rectangle coordinates of object for given image

        param: image - image to process

        param: scale_factor - scale factor used for object detection

        param: min_neighbors - minimum number of parameters considered during object detection

        param: min_size - minimum size of bounding box for object detected

        '''

        rects=self.objectCascade.detectMultiScale(image, scaleFactor=scale_factor, minNeighbors=min_neighbors,

                                                  minSize=min_size)

        return rects
FACE_DETECTION_FOLDER = '/kaggle/input/haar-cascades-for-face-detection'

frontal_cascade_path = os.path.join(FACE_DETECTION_FOLDER, 'haarcascade_frontalface_default.xml')

front_detector = ObjectDetector(frontal_cascade_path)

profile_cascade_path = os.path.join(FACE_DETECTION_FOLDER, 'haarcascade_profileface.xml')

profile_detector = ObjectDetector(profile_cascade_path)
def count_faces(video_filename):

    reader = cv2.VideoCapture(video_filename)

    nFrames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    nDetected = 0

    for i in range(nFrames):

        _, image = reader.read()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        n_profile = 0

        n_front = len(front_detector.detect(gray))

        if (n_front==0):

            n_profile = len(profile_detector.detect(gray))

        if n_front > 0 or n_profile > 0:

            nDetected += 1

    reader.release()

    return nDetected / nFrames
n_faces_real_videos = []

n_faces_fake_videos = []



for dir_name, _, file_names in os.walk('../input/deepfake-detection-challenge/train_sample_videos/'):

    for filename in file_names:

        if filename.endswith('.mp4'):

            print(filename)

            video_filename = os.path.join(dir_name, filename)

            detection_ratio = count_faces(video_filename)

            if train_sample_metadata.loc[filename]['label'] == 'REAL':

                n_faces_real_videos.append(detection_ratio)

            else:

                n_faces_fake_videos.append(detection_ratio)
# Plot an histogram of detections.

plt.figure(1)

plt.hist(n_faces_real_videos, bins=21, range=(0, 1))

plt.title('Face detection ratio of real videos')

plt.xlabel('Number of videos')

plt.ylabel('Detection ratio')

plt.figure(2)

plt.hist(n_faces_fake_videos, bins=20, range=(0, 1))

plt.title('Face detection ratio of fake videos')

plt.xlabel('Number of videos')

plt.ylabel('Detection ratio')

plt.show()
