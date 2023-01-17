TRAIN_DIR = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'



BATCH_SIZE = 1

SCALE = 0.25

N_FRAMES = None # The number of frames extracted from each video, 'None' means get all available frames
# Install facenet-pytorch

!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.0.0-py3-none-any.whl



from facenet_pytorch.models.inception_resnet_v1 import get_torch_home

torch_home = get_torch_home()



# Copy model checkpoints to torch cache so they are loaded automatically by the package

!mkdir -p $torch_home/checkpoints/

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth $torch_home/checkpoints/vggface2_DG3kwML46X.pt

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth $torch_home/checkpoints/vggface2_G5aNV2VSMn.pt
import os

import glob

import json

import torch

import cv2

from PIL import Image

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

from facenet_pytorch import MTCNN, InceptionResnetV1



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f'Running on device: {device}')
# Source: https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch

class DetectionPipeline:

    """Pipeline class for detecting faces in the frames of a video file."""

    

    def __init__(self, detector, n_frames=None, batch_size=60, resize=None):

        """Constructor for DetectionPipeline class.

        

        Keyword Arguments:

            n_frames {int} -- Total number of frames to load. These will be evenly spaced

                throughout the video. If not specified (i.e., None), all frames will be loaded.

                (default: {None})

            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})

            resize {float} -- Fraction by which to resize frames from original prior to face

                detection. A value less than 1 results in downsampling and a value greater than

                1 result in upsampling. (default: {None})

        """

        self.detector = detector

        self.n_frames = n_frames

        self.batch_size = batch_size

        self.resize = resize

    

    def __call__(self, filename):

        """Load frames from an MP4 video and detect faces.



        Arguments:

            filename {str} -- Path to video.

        """

        # Create video reader and find length

        v_cap = cv2.VideoCapture(filename)

        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))



        # Pick 'n_frames' evenly spaced frames to sample

        if self.n_frames is None:

            sample = np.arange(0, v_len)

        else:

            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)



        # Loop through frames

        faces = []

        frames = []

        for j in range(v_len):

            success = v_cap.grab()

            if j in sample:

                # Load frame

                success, frame = v_cap.retrieve()

                if not success:

                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = Image.fromarray(frame)

                

                # Resize frame to desired size

                if self.resize is not None:

                    frame = frame.resize([int(d * self.resize) for d in frame.size])

                frames.append(frame)



                # When batch is full, detect faces and reset frame list

                if len(frames) % self.batch_size == 0 or j == sample[-1]:

                    faces.extend(self.detector(frames))

                    frames = []



        v_cap.release()



        return faces
# Source: https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch

def process_faces(faces, feature_extractor):

    # Filter out frames without faces

    faces = [f for f in faces if f is not None]

    if len(faces) == 0:

        return None

    faces = torch.cat(faces).to(device)



    # Generate facial feature vectors using a pretrained model

    embeddings = feature_extractor(faces)



    # Calculate centroid for video and distance of each face's feature vector from centroid

    centroid = embeddings.mean(dim=0)

    x = (embeddings - centroid).norm(dim=1).cpu().numpy()

    

    return x
# Load face detector

face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()



# Load facial recognition model

feature_extractor = InceptionResnetV1(pretrained='vggface2', device=device).eval()



# Define face detection pipeline

detection_pipeline = DetectionPipeline(detector=face_detector, n_frames=N_FRAMES, batch_size=BATCH_SIZE, resize=SCALE)
# Get the paths of all train videos

all_train_videos = glob.glob(os.path.join(TRAIN_DIR, '*.mp4'))



# Get path of metadata.json

metadata_path = TRAIN_DIR + 'metadata.json'



# Get metadata

with open(metadata_path, 'r') as f:

    metadata = json.load(f)
df = pd.DataFrame(columns=['filename', 'distance', 'label'])



with torch.no_grad():

    for path in tqdm(all_train_videos):

        file_name = path.split('/')[-1]



        # Detect all faces occur in the video

        faces = detection_pipeline(path)

        

        # Calculate the distances of all faces' feature vectors to the centroid

        distances = process_faces(faces, feature_extractor)

        if distances is None:

            continue



        for distance in distances:

            row = [

                file_name,

                distance,

                1 if metadata[file_name]['label'] == 'FAKE' else 0

            ]



            # Append a new row at the end of the data frame

            df.loc[len(df)] = row
df.head()
df.to_csv('train.csv', index=False)