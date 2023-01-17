!pip install ../input/facenetpytorchmodel/facenet_pytorch-2.0.1-py3-none-any.whl
from facenet_pytorch import MTCNN, InceptionResnetV1

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

from pathlib import Path

import cv2

from PIL import Image

import matplotlib.pyplot as plt

import torch
IS_TEST = True

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def read_video(video_path, start_frame=0, end_frame=None, use_pbar=False):

    reader = cv2.VideoCapture(video_path)

    fps = reader.get(cv2.CAP_PROP_FPS)

    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:

        end_frame = num_frames

    pbar = tqdm(total=end_frame-start_frame, desc="Reading frames") if use_pbar else None

    frame_num = 0

    frames = []

    while reader.isOpened():

        _, img = reader.read()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:

            break

        frame_num += 1

        if frame_num < start_frame:

            continue

        frames.append(img)

        if use_pbar:

            pbar.update(1)

        if frame_num >= end_frame:

            break

    return frames



def denormalize(img):

    return ((img + 1.) * 127.5).astype(np.uint8)



def crop_faces(imgs, save_path=None):

    for img in imgs:

        img_cropped = mtcnn(Image.fromarray(img))

        img_cropped = img_cropped.permute(1, 2, 0).numpy()

        img_cropped = denormalize(img_cropped)

        plt.subplot(121)

        plt.imshow(img)

        plt.subplot(122)

        plt.imshow(img_cropped)

        plt.show()

        plt.close()
submission = pd.read_csv("../input/deepfake-detection-challenge/sample_submission.csv")
if IS_TEST:

    submission = submission.iloc[:10]
mtcnn = MTCNN(image_size=224, margin=20, device=DEVICE)
%%time



for video_fn in tqdm(submission['filename'].unique()):

    video_path = f'../input/deepfake-detection-challenge/test_videos/{video_fn}'

    frames = read_video(video_path, start_frame=0, end_frame=10, use_pbar=True)

    crop_faces(frames)