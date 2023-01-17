

#!pip install facenet-pytorch

!pip install /kaggle/input/pytorch/facenet_pytorch-2.2.8-py3-none-any.whl

from facenet_pytorch.models.inception_resnet_v1 import get_torch_home

torch_home = get_torch_home()

# Copy model checkpoints to torch cache so they are loaded automatically by the package

!mkdir -p $torch_home/checkpoints/

!pwd

!cp /kaggle/input/pytorchf/20180402-114759-vggface2-logits.pth/20180402-114759-vggface2-logits.pth $torch_home/checkpoints/vggface2_DG3kwML46X.pt

!cp /kaggle/input/pytorchf/20180402-114759-vggface2-features.pth/20180402-114759-vggface2-features.pth $torch_home/checkpoints/vggface2_G5aNV2VSMn.pt
#Import Packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

from skimage.color import rgb2gray

import cv2

import cv2 as cv

import os

import matplotlib.pylab as plt

train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'

import matplotlib.pyplot as plt

%matplotlib inline

from scipy import ndimage

# See github.com/timesler/facenet-pytorch:

from facenet_pytorch import MTCNN, InceptionResnetV1

import os

import sys

import random

import math

import numpy as np

import skimage.io

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import glob as glob

filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')

import torch  

import time

# See github.com/timesler/facenet-pytorch:

from facenet_pytorch import MTCNN, InceptionResnetV1 , extract_face

from tqdm.notebook import tqdm



#Load Data

DATA_FOLDER = '/kaggle/input/deepfake-detection-challenge'

TRAIN_SAMPLE_FOLDER = 'train_sample_videos'

TEST_FOLDER = 'test_videos'



print(f"Train samples: {len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")

print(f"Test samples: {len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")
train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))

ext_dict = []

for file in train_list:

    file_ext = file.split('.')[1]

    if (file_ext not in ext_dict):

        ext_dict.append(file_ext)

print(f"Extensions: {ext_dict}")    
for file_ext in ext_dict:

    print(f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")
test_list = list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))

ext_dict = []

for file in test_list:

    file_ext = file.split('.')[1]

    if (file_ext not in ext_dict):

        ext_dict.append(file_ext)

print(f"Extensions: {ext_dict}")

for file_ext in ext_dict:

    print(f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")
json_file = [file for file in train_list if  file.endswith('json')][0]

print(f"JSON file: {json_file}")
def get_meta_from_json(path):

    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))

    df = df.T

    return df



meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER)

meta_train_df.head()
meta_train_df.groupby('label')['label'].count().plot(figsize=(15, 5), kind='bar', title='Distribution of Labels in the Training Set')

plt.show()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(meta_train_df)
missing_data(meta_train_df.loc[meta_train_df.label=='REAL'])
def unique_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    uniques = []

    for col in data.columns:

        unique = data[col].nunique()

        uniques.append(unique)

    tt['Uniques'] = uniques

    return(np.transpose(tt))
unique_values(meta_train_df)
def most_frequent_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    items = []

    vals = []

    for col in data.columns:

        itm = data[col].value_counts().index[0]

        val = data[col].value_counts().values[0]

        items.append(itm)

        vals.append(val)

    tt['Most frequent item'] = items

    tt['Frequence'] = vals

    tt['Percent from total'] = np.round(vals / total * 100, 3)

    return(np.transpose(tt))
most_frequent_values(meta_train_df)
def plot_count(feature, title, df, size=1):

    '''

    Plot count of classes / feature

    param: feature - the feature to analyze

    param: title - title to add to the graph

    param: df - dataframe from which we plot feature's classes distribution 

    param: size - default 1.

    '''

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()    
plot_count('split', 'split (train)', meta_train_df)
plot_count('label', 'label (train)', meta_train_df)
meta = np.array(list(meta_train_df.index))

storage = np.array([file for file in train_list if  file.endswith('mp4')])

print(f"Metadata: {meta.shape[0]}, Folder: {storage.shape[0]}")

print(f"Files in metadata and not in folder: {np.setdiff1d(meta,storage,assume_unique=False).shape[0]}")

print(f"Files in folder and not in metadata: {np.setdiff1d(storage,meta,assume_unique=False).shape[0]}")
fake_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='FAKE'].sample(3).index)

fake_train_sample_video
def display_image_from_video(video_path):

    '''

    input: video_path - path for video

    process:

    1. perform a video capture from the video

    2. read the image

    3. display the image

    '''

    capture_image = cv2.VideoCapture(video_path) 

    ret, frame = capture_image.read()

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ax.imshow(frame)
for video_file in fake_train_sample_video:

    display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))
real_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='REAL'].sample(3).index)

real_train_sample_video
for video_file in real_train_sample_video:

    display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))
meta_train_df['original'].value_counts()[0:5]
def display_image_from_video_list(video_path_list, video_folder=TRAIN_SAMPLE_FOLDER):

    '''

    input: video_path_list - path for video

    process:

    0. for each video in the video path list

        1. perform a video capture from the video

        2. read the image

        3. display the image

    '''

    plt.figure()

    fig, ax = plt.subplots(2,3,figsize=(16,8))

    # we only show images extracted from the first 6 videos

    for i, video_file in enumerate(video_path_list[0:6]):

        video_path = os.path.join(DATA_FOLDER, video_folder,video_file)

        capture_image = cv2.VideoCapture(video_path) 

        ret, frame = capture_image.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ax[i//3, i%3].imshow(frame)

        ax[i//3, i%3].set_title(f"Video: {video_file}")

        ax[i//3, i%3].axis('on')
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='meawmsgiti.mp4'].index)

display_image_from_video_list(same_original_fake_train_sample_video)
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='qeumxirsme.mp4'].index)

display_image_from_video_list(same_original_fake_train_sample_video)
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='atvmxvwyns.mp4'].index)

display_image_from_video_list(same_original_fake_train_sample_video)
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='qeumxirsme.mp4'].index)

display_image_from_video_list(same_original_fake_train_sample_video)
test_videos = pd.DataFrame(list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER))), columns=['video'])
test_videos.head()
display_image_from_video(os.path.join(DATA_FOLDER, TEST_FOLDER, test_videos.iloc[0].video))
display_image_from_video_list(test_videos.sample(6).video, TEST_FOLDER)
fake_videos = list(meta_train_df.loc[meta_train_df.label=='FAKE'].index)
from IPython.display import HTML

from base64 import b64encode



def play_video(video_file, subset=TRAIN_SAMPLE_FOLDER):

    '''

    Display video

    param: video_file - the name of the video file to display

    param: subset - the folder where the video file is located (can be TRAIN_SAMPLE_FOLDER or TEST_Folder)

    '''

    video_url = open(os.path.join(DATA_FOLDER, subset,video_file),'rb').read()

    data_url = "data:video/mp4;base64," + b64encode(video_url).decode()

    return HTML("""<video width=500 controls><source src="%s" type="video/mp4"></video>""" % data_url)
play_video(fake_videos[0])
play_video(fake_videos[1])
play_video(fake_videos[2])
play_video(fake_videos[220])
import cv2 as cv

import os

import matplotlib.pylab as plt

train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'

fig, ax = plt.subplots(1,1, figsize=(15, 15))

train_video_files = [train_dir + x for x in os.listdir(train_dir)]

# video_file = train_video_files[30]

video_file = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/afoovlsmtx.mp4'

cap = cv.VideoCapture(video_file)

success, image = cap.read()

image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

cap.release()   

ax.imshow(image)

ax.xaxis.set_visible(False)

ax.yaxis.set_visible(False)

ax.title.set_text(f"FRAME 0: {video_file.split('/')[-1]}")

plt.grid(False)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f'Running on device: {device}')
# Load face detector

mtcnn = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()



# Load facial recognition model

resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
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





def process_faces(faces, resnet):

    # Filter out frames without faces

    faces = [f for f in faces if f is not None]

    faces = torch.cat(faces).to(device)



    # Generate facial feature vectors using a pretrained model

    embeddings = resnet(faces)



    # Calculate centroid for video and distance of each face's feature vector from centroid

    centroid = embeddings.mean(dim=0)

    x = (embeddings - centroid).norm(dim=1).cpu().numpy()

    

    return x
   # Define face detection pipeline

detection_pipeline = DetectionPipeline(detector=mtcnn, batch_size=60, resize=0.25)



# Get all test videos

filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')



X = []

start = time.time()

n_processed = 0

with torch.no_grad():

    for i, filename in tqdm(enumerate(filenames), total=len(filenames)):

        try:

            # Load frames and find faces

            faces = detection_pipeline(filename)

            

            # Calculate embeddings

            X.append(process_faces(faces, resnet))



        except KeyboardInterrupt:

            print('\nStopped.')

            break



        except Exception as e:

            print(e)

            X.append(None)

        

        n_processed += len(faces)

        print(f'Frames per second (load+detect+embed): {n_processed / (time.time() - start):6.3}\r', end='')
bias = -0.4

weight = 0.068235746



submission = []

for filename, x_i in zip(filenames, X):

    if x_i is not None and len(x_i) == 10:

        prob = 1 / (1 + np.exp(-(bias + (weight * x_i).sum())))

    else:

        prob = 0.5

    submission.append([os.path.basename(filename), prob])
submission = pd.DataFrame(submission, columns=['filename', 'label'])

submission.sort_values('filename').to_csv('submission.csv', index=False)
plt.hist(submission.label, 20)

plt.show()

submission