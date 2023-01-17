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

#!pip install mtcnn


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from keras.datasets import mnist

from keras.layers import Input, Dense, Reshape, Flatten, Dropout

from keras.layers import LSTM

from keras.layers import TimeDistributed

from keras.layers import BatchNormalization, Activation, ZeroPadding2D

from keras.layers import LeakyReLU

from keras.models import Sequential, Model

from keras.optimizers import Adam

from mtcnn import MTCNN

from smart_open import smart_open

import matplotlib.pyplot as plt

import cv2

from io import BytesIO

import numpy as np

import boto3

import pandas as pd

from PIL import Image

from tqdm.notebook import tqdm

import time

from keras.models import model_from_json



import sys
from keras.models import model_from_json

json_file = open('../input/models/model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()



loaded_model1 = model_from_json(loaded_model_json)

json_file = open('../input/models/model2.json', 'r')

loaded_model_json2 = json_file.read()

json_file.close()

loaded_model2 = model_from_json(loaded_model_json2)



json_file = open('../input/models/model3.json', 'r')

loaded_model_json3 = json_file.read()

json_file.close()

loaded_model3 = model_from_json(loaded_model_json3)

def detect_mtcnn(detector, images):

    faces =[]

    for image in images:

        boxes = detector.detect_faces(image)

        box = boxes[0]['box']

        face = image[box[1]:box[3]+box[1], box[0]:box[2]+box[0]]

        faces.append(face)



    return faces
def timer(detector, detect_fn, images, *args):

                    start = time.time()

                    faces = detect_fn(detector, images, *args)

                    elapsed = time.time() - start

                    return faces, elapsed  
DATA_FOLDER = '../input/deepfake-detection-challenge'

TRAIN_SAMPLE_FOLDER = 'train_sample_videos'

TEST_FOLDER = 'test_videos'

def display_image_from_video(video_path):

    '''

    input: video_path - path for video

    process:

    1. perform a video capture from the video

    2. read the image

    3. display the image

    '''

    capture_image = cv.VideoCapture(video_path) 

    ret, frame = capture_image.read()

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111)

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    ax.imshow(frame)

display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, 'aettqgevhz.mp4'))
reader = cv2.VideoCapture('/kaggle/input/deepfake-detection-challenge/train_sample_videos/aettqgevhz.mp4')

images_540_960 = []

for i in tqdm(range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT)))):

    _, image = reader.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images_540_960.append(cv2.resize(image, (960, 540)))

reader.release()

images_540_960 = np.stack(images_540_960)
from mtcnn import MTCNN

detector = MTCNN()

final=[]

total=299

X_train=[]

faces, elapsed = timer(detector, detect_mtcnn, images_540_960)

final.append(faces)

final=[[cv2.resize(face, (16, 16)) for face in x] for x in final]

final=[np.array(x) for x in final]



if final[0].shape[0]<total:

    length=total-final[0].shape[0]

    np.append(final[0],np.zeros((length,16,16,3)))

else:

    final[0]=final[0][:total,:,:,:]

    print('coorected')

try:

    X_train.extend(final)

    print('tried')

except:

    X_train= final

X_trainy=np.expand_dims(X_train[0], axis=0)

loaded_model3.predict(X_trainy)