import os, sys, time

import cv2

import numpy as np

import pandas as pd



import torch

import torch.nn as nn

import torch.nn.functional as F



%matplotlib inline

import matplotlib.pyplot as plt
train_dir = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/"



train_videos = sorted([x for x in os.listdir(train_dir) if x[-4:] == ".mp4"])

len(train_videos)
print("PyTorch version:", torch.__version__)

print("CUDA version:", torch.version.cuda)

print("cuDNN version:", torch.backends.cudnn.version())
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gpu
import sys

sys.path.insert(0, "/kaggle/input/blazeface-pytorch")

sys.path.insert(0, "/kaggle/input/deepfakes-inference-demo")
from blazeface import BlazeFace

facedet = BlazeFace().to(gpu)

facedet.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")

facedet.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")

_ = facedet.train(False)
from helpers.read_video_1 import VideoReader

from helpers.face_extract_1 import FaceExtractor



frames_per_video = 17



video_reader = VideoReader()

video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)

face_extractor = FaceExtractor(video_read_fn, facedet)
input_size = 224
from torchvision.transforms import Normalize



mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]

normalize_transform = Normalize(mean, std)
def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):

    h, w = img.shape[:2]

    if w > h:

        h = h * size // w

        w = size

    else:

        w = w * size // h

        h = size



    resized = cv2.resize(img, (w, h), interpolation=resample)

    return resized





def make_square_image(img):

    h, w = img.shape[:2]

    size = max(h, w)

    t = 0

    b = size - h

    l = 0

    r = size - w

    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
np.random.choice(train_videos)
faces = face_extractor.process_video(os.path.join(train_dir,np.random.choice(train_videos)))

type(faces)
len(faces)
ex1=np.array(faces[0]["faces"][0])
resized_image=isotropically_resize_image(ex1,input_size)

resized_image=make_square_image(ex1)

plt.imshow(resized_image)
# plt.savefig('./face1.jpg')

cv2.imwrite('./face1.jpg',resized_image)