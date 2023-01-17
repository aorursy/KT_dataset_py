from IPython.display import Image
Image('../input/deepfake-kernel-data/google_cloud_compute_engine_launch_vm.png')
Image('../input/deepfake-kernel-data/google_cloud_vm.png')
!ls ../input/deepfake-kernel-data
Image('../input/deepfake-kernel-data/lr_15e-2_epochs_42_patience_5.png')
Image('../input/deepfake-kernel-data/lr_2e-3_epochs_10_patience_5.png')
Image('../input/deepfake-kernel-data/lr_2e-3_epochs_20_patience_5.png')
Image('../input/deepfake-kernel-data/lr_4e-3_epochs_12_patience_2.png')
Image('../input/deepfake-kernel-data/lr_4e-3_epochs_30_patience_2.png')
Image('../input/deepfake-kernel-data/google_cloud_vm_deepfake_training_screenshot.png')
import os, sys, time

import cv2

import numpy as np

import pandas as pd



import torch

import torch.nn as nn

import torch.nn.functional as F



%matplotlib inline

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")
test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"



test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])

frame_h = 5

frame_l = 5

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



frames_per_video = 64 #frame_h * frame_l

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
def predict_on_video(video_path, batch_size):

    try:

        # Find the faces for N frames in the video.

        faces = face_extractor.process_video(video_path)



        # Only look at one face per frame.

        face_extractor.keep_only_best_face(faces)

        

        if len(faces) > 0:

            # NOTE: When running on the CPU, the batch size must be fixed

            # or else memory usage will blow up. (Bug in PyTorch?)

            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)



            # If we found any faces, prepare them for the model.

            n = 0

            for frame_data in faces:

                for face in frame_data["faces"]:

                    # Resize to the model's required input size.

                    # We keep the aspect ratio intact and add zero

                    # padding if necessary.                    

                    resized_face = isotropically_resize_image(face, input_size)

                    resized_face = make_square_image(resized_face)



                    if n < batch_size:

                        x[n] = resized_face

                        n += 1

                    else:

                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))

                    

                    # Test time augmentation: horizontal flips.

                    # TODO: not sure yet if this helps or not

                    #x[n] = cv2.flip(resized_face, 1)

                    #n += 1



            if n > 0:

                x = torch.tensor(x, device=gpu).float()



                # Preprocess the images.

                x = x.permute((0, 3, 1, 2))



                for i in range(len(x)):

                    x[i] = normalize_transform(x[i] / 255.)



                # Make a prediction, then take the average.

                with torch.no_grad():

                    y_pred = model(x)

                    y_pred = torch.sigmoid(y_pred.squeeze())

                    return y_pred[:n].mean().item()



    except Exception as e:

        print("Prediction error on video %s: %s" % (video_path, str(e)))



    return 0.5
from concurrent.futures import ThreadPoolExecutor



def predict_on_video_set(videos, num_workers):

    def process_file(i):

        filename = videos[i]

        y_pred = predict_on_video(os.path.join(test_dir, filename), batch_size=frames_per_video)

        return y_pred



    with ThreadPoolExecutor(max_workers=num_workers) as ex:

        predictions = ex.map(process_file, range(len(videos)))



    return list(predictions)
speed_test = False  # you have to enable this manually
if speed_test:

    start_time = time.time()

    speedtest_videos = test_videos[:5]

    predictions = predict_on_video_set(speedtest_videos, num_workers=4)

    elapsed = time.time() - start_time

    print("Elapsed %f sec. Average per video: %f sec." % (elapsed, elapsed / len(speedtest_videos)))
predictions = predict_on_video_set(test_videos, num_workers=4)
submission_df_resnext = pd.DataFrame({"filename": test_videos, "label": predictions})

submission_df_resnext.to_csv("submission_resnext.csv", index=False)
!pip install ../input/deepfake-xception-trained-model/pytorchcv-0.0.55-py2.py3-none-any.whl --quiet
test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"



test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])

len(test_videos)
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



frames_per_video = 64 # originally 4



video_reader = VideoReader()

video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)

face_extractor = FaceExtractor(video_read_fn, facedet)
input_size = 150
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
!ls ../input/deepfake-xception-trained-model
!ls ../input/deepfake-kernel-data
from pytorchcv.model_provider import get_model

model = get_model("xception", pretrained=False)

model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer



class Pooling(nn.Module):

  def __init__(self):

    super(Pooling, self).__init__()

    

    self.p1 = nn.AdaptiveAvgPool2d((1,1))

    self.p2 = nn.AdaptiveMaxPool2d((1,1))



  def forward(self, x):

    x1 = self.p1(x)

    x2 = self.p2(x)

    return (x1+x2) * 0.5



model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)))



class Head(torch.nn.Module):

  def __init__(self, in_f, out_f):

    super(Head, self).__init__()

    

    self.f = nn.Flatten()

    self.l = nn.Linear(in_f, 512)

    self.d = nn.Dropout(0.5)

    self.o = nn.Linear(512, out_f)

    self.b1 = nn.BatchNorm1d(in_f)

    self.b2 = nn.BatchNorm1d(512)

    self.r = nn.ReLU()



  def forward(self, x):

    x = self.f(x)

    x = self.b1(x)

    x = self.d(x)



    x = self.l(x)

    x = self.r(x)

    x = self.b2(x)

    x = self.d(x)



    out = self.o(x)

    return out



class FCN(torch.nn.Module):

  def __init__(self, base, in_f):

    super(FCN, self).__init__()

    self.base = base

    self.h1 = Head(in_f, 1)

  

  def forward(self, x):

    x = self.base(x)

    return self.h1(x)



net = []

model = FCN(model, 2048)

model = model.cuda()

model.load_state_dict(torch.load('../input/deepfake-kernel-data/model_50epochs_lr0001_patience5_factor01_batchsize32.pth')) # new, updated

net.append(model)
def predict_on_video(video_path, batch_size):

    try:

        # Find the faces for N frames in the video.

        faces = face_extractor.process_video(video_path)



        # Only look at one face per frame.

        face_extractor.keep_only_best_face(faces)

        

        if len(faces) > 0:

            # NOTE: When running on the CPU, the batch size must be fixed

            # or else memory usage will blow up. (Bug in PyTorch?)

            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)



            # If we found any faces, prepare them for the model.

            n = 0

            for frame_data in faces:

                for face in frame_data["faces"]:

                    # Resize to the model's required input size.

                    # We keep the aspect ratio intact and add zero

                    # padding if necessary.                    

                    resized_face = isotropically_resize_image(face, input_size)

                    resized_face = make_square_image(resized_face)



                    if n < batch_size:

                        x[n] = resized_face

                        n += 1

                    else:

                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))

                    

                    # Test time augmentation: horizontal flips.

                    # TODO: not sure yet if this helps or not

                    #x[n] = cv2.flip(resized_face, 1)

                    #n += 1



            if n > 0:

                x = torch.tensor(x, device=gpu).float()



                # Preprocess the images.

                x = x.permute((0, 3, 1, 2))



                for i in range(len(x)):

                    x[i] = normalize_transform(x[i] / 255.)

#                     x[i] = x[i] / 255.



                # Make a prediction, then take the average.

                with torch.no_grad():

                    y_pred = model(x)

                    y_pred = torch.sigmoid(y_pred.squeeze())

                    return y_pred[:n].mean().item()



    except Exception as e:

        print("Prediction error on video %s: %s" % (video_path, str(e)))



    return 0.5
from concurrent.futures import ThreadPoolExecutor



def predict_on_video_set(videos, num_workers):

    def process_file(i):

        filename = videos[i]

        y_pred = predict_on_video(os.path.join(test_dir, filename), batch_size=frames_per_video)

        return y_pred



    with ThreadPoolExecutor(max_workers=num_workers) as ex:

        predictions = ex.map(process_file, range(len(videos)))



    return list(predictions)
speed_test = False
if speed_test:

    start_time = time.time()

    speedtest_videos = test_videos[:5]

    predictions = predict_on_video_set(speedtest_videos, num_workers=4)

    elapsed = time.time() - start_time

    print("Elapsed %f sec. Average per video: %f sec." % (elapsed, elapsed / len(speedtest_videos)))
%%time

model.eval()

predictions = predict_on_video_set(test_videos, num_workers=4)
submission_df_xception = pd.DataFrame({"filename": test_videos, "label": predictions})

submission_df_xception.to_csv("submission_xception.csv", index=False)
submission_df_resnext.head()
submission_df_xception.head()
submission_df = pd.DataFrame({"filename": test_videos})

submission_df["label"] = 0.5*submission_df_resnext["label"] + 0.5*submission_df_xception["label"]
submission_df.to_csv("submission.csv", index=False)