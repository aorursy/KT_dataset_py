%%capture

# Install facenet-pytorch

!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-1.0.1-py3-none-any.whl



# Copy model checkpoints to torch cache so they are loaded automatically by the package

!mkdir -p /tmp/.cache/torch/checkpoints/

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth /tmp/.cache/torch/checkpoints/vggface2_DG3kwML46X.pt

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth /tmp/.cache/torch/checkpoints/vggface2_G5aNV2VSMn.pt



# Install ffmpeg

! tar xvf ../input/ffmpeg-static-build/ffmpeg-git-amd64-static.tar.xz
import os

import glob

import torch

import cv2

from PIL import Image

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from tqdm import tqdm_notebook



# See github.com/timesler/facenet-pytorch:

from facenet_pytorch import MTCNN, InceptionResnetV1



device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Running on device: {device}')
# Load face detector

mtcnn = MTCNN(device='cuda').eval()



# Load facial recognition model

resnet = InceptionResnetV1(pretrained='vggface2', num_classes=2, device=device).eval()
# Get all test videos

filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')



# Number of frames to sample (evenly spaced) from each video

n_frames = 10



X = []

with torch.no_grad():

    for i, filename in enumerate(filenames):

        print(f'Processing {i+1:5n} of {len(filenames):5n} videos\r', end='')

        

        try:

            # Create video reader and find length

            v_cap = cv2.VideoCapture(filename)

            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            

            # Pick 'n_frames' evenly spaced frames to sample

            sample = np.linspace(0, v_len - 1, n_frames).round().astype(int)

            imgs = []

            for j in range(v_len):

                success = v_cap.grab()

                if j in sample:

                    success, vframe = v_cap.retrieve()

                    vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)

                    imgs.append(Image.fromarray(vframe))

            v_cap.release()

            

            # Pass image batch to MTCNN as a list of PIL images

            faces = mtcnn(imgs)

            

            # Filter out frames without faces

            faces = [f for f in faces if f is not None]

            faces = torch.stack(faces).cuda()

            

            # Generate facial feature vectors using a pretrained model

            embeddings = resnet(faces).cuda()

            

            # Calculate centroid for video and distance of each face's feature vector from centroid

            centroid = embeddings.mean(dim=0)

            X.append((embeddings - centroid).norm(dim=1).cpu().numpy())

        except KeyboardInterrupt:

            raise Exception("Stopped.")
np.save('X.npy',X)
bias = -0.2942

weight = 0.068235746



submission = []

for filename, x_i in zip(filenames, X):

    if x_i is not None and len(x_i) == 10:

        prob = 1 / (1 + np.exp(-(bias + (weight * x_i).sum())))

    else:

        prob = 0.5

    submission.append([os.path.basename(filename), prob])
submission = pd.DataFrame(submission, columns=['filename', 'label'])
sub = pd.read_csv('../input/deepfake-detection-challenge/sample_submission.csv')
result_map=dict(zip(submission.filename,submission.label))

sub['label']=sub['filename'].map(result_map)

sub.label = np.where(sub['label']>0.5,1,0)
sub.to_csv('submission.csv',index=False)