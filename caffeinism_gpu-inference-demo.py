video_batch_size = 10

frames_per_video = 17

input_size = 224

test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"
import os, sys, time

import cv2

import numpy as np

import pandas as pd



import torch

import torch.nn as nn

import torch.nn.functional as F



%matplotlib inline

import matplotlib.pyplot as plt
test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])

len(test_videos)
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



video_reader = VideoReader()

video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)

face_extractor = FaceExtractor(video_read_fn, facedet)
import torch.nn as nn

import torchvision.models as models



class MyResNeXt(models.resnet.ResNet):

    def __init__(self, training=True):

        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,

                                        layers=[3, 4, 6, 3], 

                                        groups=32, 

                                        width_per_group=4)

        self.fc = nn.Linear(2048, 1)
checkpoint = torch.load("/kaggle/input/deepfakes-inference-demo/resnext.pth", map_location=gpu)



model = MyResNeXt().to(gpu)

model.load_state_dict(checkpoint)

_ = model.eval()



del checkpoint
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
import torchvision.transforms as transforms



transform = transforms.Compose([

    lambda x: isotropically_resize_image(x, input_size),

    make_square_image,

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485, 0.456, 0.406],

        [0.229, 0.224, 0.225]

    ),

])
import collections, itertools



def predict_on_video(video_paths):

    # Find the faces for N frames in the video.

    # Only look at one face per frame.

    faces = face_extractor.process_videos(test_dir, video_paths, range(len(video_paths)))

    face_extractor.keep_only_best_face(faces)



    videos = collections.defaultdict(list)

    for face in faces:

        if len(face['faces']) > 0:

            videos[video_paths[face['video_idx']]].append(transform(face['faces'][0]))



    data = [

        (video_path, 

         torch.stack(videos[video_path]), 

         len(videos[video_path])

        ) for video_path in video_paths if len(videos[video_path]) > 0

    ]

    

    unknown_video_path = [video_path for video_path in video_paths if len(videos[video_path]) == 0]



    if len(data) > 0:

        known_video_path, video_tensors, video_lengths = zip(*data)

        video_batch = torch.cat(video_tensors).to(gpu)



        with torch.no_grad():

            y_pred = model(video_batch)

            y_pred = torch.sigmoid(y_pred.squeeze())

        video_pred = [it.mean().item() for it in y_pred.split(video_lengths)]

    else:

        known_video_path, video_pred = tuple(), tuple()



    known_answer = zip(known_video_path, video_pred)

    unknown_answer = zip(unknown_video_path, [0.5] * len(unknown_video_path))



    return itertools.chain(known_answer, unknown_answer)
def chunks(lst, n):

    for i in range(0, len(lst), n):

        yield lst[i:i + n]    



def predict_on_video_set(video_paths):

    predictions = [predict_on_video(batch) for batch in chunks(video_paths, video_batch_size)]

    return itertools.chain(*predictions)
speed_test = False  # you have to enable this manually

if speed_test:

    start_time = time.time()

    predictions = predict_on_video_set(test_videos[:16])

    print(time.time() - start_time)
results = predict_on_video_set(test_videos)
filenames, labels = zip(*results)

submission_df = pd.DataFrame({"filename": filenames, "label": labels})

submission_df.to_csv("submission.csv", index=False)
%matplotlib inline

submission_df.plot.hist(bins=11)