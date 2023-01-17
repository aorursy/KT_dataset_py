import numpy as np

import pandas as pd

import json

import cv2

import os

import matplotlib.pylab as plt

from random import shuffle

from tqdm import tqdm

import pickle

import keras

from keras import optimizers

import tensorflow

from keras.preprocessing import image

import math

import matplotlib.pyplot as plt    # for plotting the images

%matplotlib inline

from keras.utils import np_utils

from keras.applications.vgg16 import preprocess_input

from sklearn.model_selection import train_test_split

from keras.applications.vgg16 import VGG16

from keras.layers import Dense, InputLayer, Dropout
from keras.applications.vgg19 import VGG19
from skimage.transform import resize

import pickle
sample_submission = pd.read_csv("../input/deepfake-detection-challenge/sample_submission.csv")

train_sample_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T



train_dir = '../input/deepfake-detection-challenge/train_sample_videos/'

train_video_filespath = [train_dir + x for x in tqdm(os.listdir(train_dir))]

train_sample_imgs = [x for x in tqdm(os.listdir(train_dir))]



test_dir = '../input/deepfake-detection-challenge/test_videos/'

test_video_filespath = [test_dir + x for x in tqdm(os.listdir(test_dir))]

test_imgs = [x for x in tqdm(os.listdir(test_dir))]



train_sample_imgs.remove('metadata.json')

train_video_filespath.remove('../input/deepfake-detection-challenge/train_sample_videos/metadata.json')



train_video_filespath = sorted(train_video_filespath)

train_sample_imgs = sorted(train_sample_imgs)

train_sample_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T

(train_sample_metadata)
Total = []

#train_sample_metadata = sorted(train_sample_metadata)

print(train_sample_metadata)

for i in range(len(train_video_filespath)):

    Total.append([train_video_filespath[i],train_sample_metadata.iloc[i,0]])
REAL = []

FAKE = []

for i in range(len(Total)):

    if(Total[i][1] == "FAKE"):

        FAKE.append(Total[i])

    else:

        REAL.append(Total[i])
import random



FAKE = random.sample(FAKE,len(REAL))
train_video_filespath = REAL + FAKE
shuffle(train_video_filespath)
X = []

y = []

detector = MTCNN()

def detect_face(img):

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    final = []

    detected_faces_raw = detector.detect_faces(img)

    if detected_faces_raw==[]:

        #print('no faces found')

        return []

    confidences=[]

    for n in detected_faces_raw:

        x,y,w,h=n['box']

        final.append([x,y,w,h])

        confidences.append(n['confidence'])

    if max(confidences)<0.7:

        return []

    max_conf_coord=final[confidences.index(max(confidences))]

    #return final

    return max_conf_coord

def crop(img,x,y,w,h):

    x-=40

    y-=40

    w+=80

    h+=80

    if x<0:

        x=0

    if y<=0:

        y=0

    return cv2.cvtColor(cv2.resize(img[y:y+h,x:x+w],(224,224)),cv2.COLOR_BGR2RGB)

def detect_video(video):

    v_cap = cv2.VideoCapture(video)

    v_cap.set(1, NUM_FRAME)

    success, vframe = v_cap.read()

    vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)

    bounding_box=detect_face(vframe)

    if bounding_box==[]:

        count=0

        current=NUM_FRAME

        while bounding_box==[] and count<MAX_SKIP:

            current+=1

            v_cap.set(1,current)

            success, vframe = v_cap.read()

            vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)

            bounding_box=detect_face(vframe)

            count+=1

        if bounding_box==[]:

            print('hi')

            return None

    x,y,w,h=bounding_box

    v_cap.release()

    return crop(vframe,x,y,w,h)



MAX_SKIP=10

NUM_FRAME=150

count=0

for video in tqdm(train_video_filespath):

    img_file=detect_video(video)

    X.append(img_file)

    y.append(train_video_filespath[i][1])
X = []

y = []

for i in tqdm(range(len(train_video_filespath))):

    count = 0

    videoFile = train_video_filespath[i][0]

    cap = cv2.VideoCapture(videoFile)

    frameRate = cap.get(5) #frame rate

    x=1

    while(cap.isOpened()):

        frameId = cap.get(1) #current frame number

        ret, frame = cap.read()

        if (ret != True):

            break

        if (frameId % math.floor(frameRate) == 0):

            frame = detectface()

            a = resize(frame, preserve_range=True, output_shape=(224,224)).astype(int)

            X.append(a)

            y.append(train_video_filespath[i][1])

    cap.release()



X = np.array(X)    # converting list to array 



for i in range(len(y)):

    if(y[i] == 'FAKE'):

        y[i] = 1

    else:

        y[i] = 0



dummy_y = np_utils.to_categorical(y)
X = preprocess_input(X, mode='tf')      # preprocessing the input data
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer
base_model19 = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer
X1 = base_model.predict(X)
!pip install mtcnn
from mtcnn import MTCNN
X1 = X1.reshape(1694,7*7*512)
train = X1/X1.max()      # centering the data
train.shape
model = keras.models.Sequential()

model.add(InputLayer((7*7*512,)))    # input layer

model.add(Dense(units=1024, activation='relu')) # hidden layer

#model.add(Dense(units=1024, activation='relu')) # hidden layer

model.add(Dense(2, activation='softmax'))    # output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

history = model.fit(train, dummy_y, epochs=100, validation_split = 0.1, batch_size = 32, verbose = 1)
plt.plot(history.history['val_accuracy'])