%%capture

!pip install --no-deps ../input/maruti/maruti-1.3.1-py3-none-any.whl

!pip install /kaggle/input/facenetmtcnn/facenet_pytorch-2.1.1-py3-none-any.whl
import os

import torch

from facenet_pytorch import MTCNN

import pandas as pd

import cv2

from tqdm import tqdm

from PIL import Image

import numpy as np

import math

from torchvision import transforms

import torchvision.models as models

from maruti.imports.ml import *

from torch.nn.utils.rnn import pack_sequence

import keras

import tensorflow
test_dir = '../input/deepfake-detection-challenge/test_videos/'

test_videos = sorted(os.listdir(test_dir))
with tensorflow.device('/gpu:0'):

    model = tensorflow.keras.models.load_model('/kaggle/input/pytorch-keras-models/Resnext101.h5')
device = torch.device("cuda")
preprocess = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize([256,256]),

    transforms.CenterCrop([224,224]),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
mtcnn = MTCNN(select_largest=False,device=device)

group_transform = mdata.group_transform['val'] # default transform

group_transform = lambda x: torch.stack(tuple(map(preprocess, x)))



def get_brightness_score(img):

    return maruti.vision.image.brightness_score(img)



def adjust_brightness(img):

    if(get_brightness_score(img) < 1.5):

        return maruti.vision.image.adjust_brightness(img,1.5)



def get_batch(path):

    frame_count, _ = mvis.vid_info(path)

    frame_idx = np.linspace(0, frame_count-1, 8, dtype=int)

    frame_list = list(mvis.get_face_frames(path, frame_idx))

    for i in frame_list:

        i = np.array(adjust_brightness(i))

    return frame_list
#SPEED TEST

speed_test = False

if speed_test:

    start = time.perf_counter()

    for vid in tqdm(test_videos[:10]):

        print(predict(test_dir+vid))

    print((time.perf_counter()-start)/10)
start = time.perf_counter()

predictions = []

for i, vid in tqdm(enumerate(test_videos)):

    if i%20==19:

        os.system(f'echo {str(i)} {predictions[-1]:.2f}')

    try:

        pred = []

        x = np.array(get_batch(test_dir+vid))

        for i in x:

            i = np.reshape(i,(1,224,224,3))

        predictions.append(np.mean(np.array(model.predict(x))))

    except Exception as e:

        print(vid+' error:'+str(e))

        predictions.append(0.5)



print((time.perf_counter()-start)/len(test_videos))
submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})

submission_df['label'] = np.clip(submission_df['label'],0.05,0.95)

submission_df.to_csv("submission.csv", index=False)