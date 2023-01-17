""" To estimate execution time of the Kernel """

import time

start = time.time()
""" Include packages """

import pandas as pd

import glob

import os

import subprocess as sp

import tqdm.notebook as tqdm

from tqdm import tqdm

from collections import defaultdict

import numpy as np

import json



!pip install /kaggle/input/mtcnn-package/mtcnn-0.1.0-py3-none-any.whl  

import cv2

from mtcnn import MTCNN
""" Read files from test folder """

test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/'

filenames=os.listdir(test_dir)

filenames.sort()

test_video_files = [test_dir + x for x in filenames]
""" Utility functions """

detector = MTCNN()



import pickle

with open('/kaggle/input/db0001/model_0001.pkl', 'rb') as f:

    model = pickle.load(f)

    

# Parameters for contrast enhancement

lookUpTable = np.empty((1,256), np.uint8)

gamma = 0.5

for i in range(256):

    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)    



def detect_face(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    final = []

    detected_faces_raw = detector.detect_faces(img)

    if detected_faces_raw == []:

        print('no faces found, skip to next frame', end='\r')

        return []

    for x in detected_faces_raw:

        x, y, w, h = x['box']

        final.append([x, y, w, h])

    return final



def crop(img, x, y, w, h):

    x -= 40

    y -= 40

    w += 80

    h += 80

    if x < 0:

        x = 0

    if y <= 0:

        y = 0

    

    frame = cv2.resize(img[y: y + h, x: x + w],(256,256),interpolation = cv2.INTER_AREA)

    return (255*(frame/frame.max())).astype('uint8')



def detect_video(video):

    cap = cv2.VideoCapture(video)

    max_skip = 0

    while True:

        ret = cap.grab()

        ret, frame = cap.retrieve()

        

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        

        #cv2.normalize(frame,  frame, 0, 255, cv2.NORM_MINMAX)

        #frame = cv2.LUT(frame, lookUpTable)

            

        bounding_box = detect_face(frame)

        if bounding_box == []:

            if(max_skip==10):

                return []

            max_skip += 1

            continue

        x, y, w, h = bounding_box[0]

        return crop(frame, x, y, w, h) 

    

def predict(frame, model):

    if frame == []:

        return []

    else:

        frame = np.expand_dims(frame, axis = 0)

        return np.around(model.predict(frame).clip(0.15,0.85).astype('float64'), decimals=2)
""" Read sample submission """

sub = pd.read_csv('/kaggle/input/deepfake-detection-challenge/sample_submission.csv')

sub.label = 0.5

sub = sub.set_index('filename',drop=False)
""" Update submission file with predcition from Keras model """

count = 0

for filename in test_video_files:

    print(count, end='\r')

    count += 1

    fn = filename.split('/')[-1]

    if detect_video(filename)==[]:

        sub.loc[fn, 'label'] = 0.5 

    else:

        pred = predict(detect_video(filename), model)

        sub.loc[fn, 'label'] = pred[0][1]  
sub.head()
""" Write submission csv """

sub.to_csv('submission.csv', index=False)
""" How long it took? """

end = time.time()  

print('Time: ', end-start)