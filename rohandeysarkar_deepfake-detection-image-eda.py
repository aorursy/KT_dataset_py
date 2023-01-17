import numpy as np

import pandas as pd

import os

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

%matplotlib inline 

import cv2 as cv

# https://www.kaggle.com/gpreda/deepfake-starter-kit
DATA_FOLDER = '../input/deepfake-detection-challenge'

TRAIN_SAMPLE_FOLDER = 'train_sample_videos'

TEST_FOLDER = 'test_videos'



print(f"Train samples:{len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")

print(f"Test samples: {len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")



FACE_DETECTION_FOLDER = '../input/haar-cascades-for-face-detection'

print(f"Face detection resources: {os.listdir(FACE_DETECTION_FOLDER)}")
# Check files type

train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))

ext_list = []

for file in train_list:

    file_ext = file.split('.')[1]

    if (file_ext not in ext_list):

        ext_list.append(file_ext)

print(f"Extensions: {ext_list}")



# count how many files with each extensions there are

for file_ext in ext_list:

    print('No. of', file_ext, ':', len([file for file in train_list if file.endswith(file_ext)]))
test_list = list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))

ext_list = []

for file in test_list:

    file_ext = file.split('.')[1]

    if (file_ext not in ext_list):

        ext_list.append(file_ext)

print(f"Extensions: {ext_list}")



for file_ext in ext_list:

    print('No. of', file_ext, ':', len([file for file in test_list if file.endswith(file_ext)]))
json_file = [file for file in train_list if file.endswith('json')][0]

print(f"json file: {json_file}")
# There is a json file(metadata.json) & we will explore this file

def get_meta_from_json(path):

    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))

    # rows and cols are diffult to see so apply transpose

    df = df.T

    return df
meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER)

meta_train_df.head()

# the index has the .mp4 file names
# Missing data

def missing_data(data):

    total = data.isnull().sum()

    percent = (total/data.isnull().count()*100)

    tp = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tp['Types'] = types

    

    return tp



missing_data(meta_train_df)
missing_data(meta_train_df.loc[meta_train_df.label=='REAL'])

# all missing original data are the one associated with REAL label

# missing_data(meta_train_df.loc[meta_train_df.label=='FAKE'])
# Unique vals

def unique_values(data):

    total = data.count()

    t = pd.DataFrame(total)

    t.columns = ['Total']

    

    uniques = []

    for col in data.columns:

        unique = data[col].nunique()

        uniques.append(unique)

    t['Uniques'] = uniques

    

    return t

    

unique_values(meta_train_df)

# We observe that original label has the same pattern for uniques values. 

# We know that we have 77 missing data (that's why total is only 323) 

# and we observe that we do have 209 unique examples.
# Most frequent originals

def most_frequent_values(data):

    total = data.count()

    t = pd.DataFrame(total)

    t.columns = ['Total']

    

    items = []

    vals = []

    for col in data.columns:

        # count unique rows -> value_counts(), taking 1st index as it's val is greater so more occurence/frequent

        itm = data[col].value_counts().index[0]

        val = data[col].value_counts().values[0]

        items.append(itm)

        vals.append(val)

    t['Most frequent item'] = items

    t['Frequence'] = vals

    t['Percent from total'] = np.round(vals / total * 100, 3)

    return t

  

most_frequent_values(meta_train_df)

# We see that most frequent label is FAKE (80.75%), 

# meawmsgiti.mp4 is the most frequent original (6 samples).
def plot_count(feature, title, df):

    f, ax = plt.subplots(1, 1, figsize=(4, 4))

    total = float(len(df))

    g = sns.countplot(x=feature, data=df, order=df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("No. and percentage of {}".format(title))

    

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x() + p.get_width()/2, height, '{:1.2f}%'.format((height/total)*100), ha="center")

    plt.show()

    

plot_count('split', 'split (train)', meta_train_df)

plot_count('label', 'label (train)', meta_train_df)

# print(meta_train_df['label'].value_counts().index[:2])
# Video data exploration

# Missing video (or meta) data

meta = np.array(list(meta_train_df.index))

storage = np.array([file for file in train_list if file.endswith('mp4')])

print(f"Metadata: {meta.shape[0]}, Folder: {storage.shape[0]}")

print(f"Files in metadata and not in folder: {np.setdiff1d(meta,storage,assume_unique=False).shape[0]}")

print(f"Files in folder and not in metadata: {np.setdiff1d(storage,meta,assume_unique=False).shape[0]}")
# Few fake videos(the index contains the .mp4 file names so .index gives the .mp4 file names)

fake_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='FAKE'].sample(3).index)

fake_train_sample_video
def display_image_from_video(video_path):

    cap_img = cv.VideoCapture(video_path)

    ret, frame = cap_img.read()

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111)

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    ax.imshow(frame)
for video_file in fake_train_sample_video:

    display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))
# Few real videos

real_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='REAL'].sample(3).index)

real_train_sample_video
for video_file in real_train_sample_video:

    display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))
# Videos with same original

meta_train_df['original'].value_counts()[0:5]
def display_image_from_video_list(video_path_list, video_folder=TRAIN_SAMPLE_FOLDER):

    plt.figure()

    fig, ax = plt.subplots(2,3,figsize=(16,8))

    # we only show images extracted from the first 6 videos

    for i, video_file in enumerate(video_path_list[0:6]):

        video_path = os.path.join(DATA_FOLDER, video_folder,video_file)

        capture_image = cv.VideoCapture(video_path) 

        ret, frame = capture_image.read()

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        ax[i//3, i%3].imshow(frame)

        ax[i//3, i%3].set_title(f"Video: {video_file}")

        ax[i//3, i%3].axis('on')
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='meawmsgiti.mp4'].index)

display_image_from_video_list(same_original_fake_train_sample_video)
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='qeumxirsme.mp4'].index)

display_image_from_video_list(same_original_fake_train_sample_video)
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='kgbkktcjxf.mp4'].index)

display_image_from_video_list(same_original_fake_train_sample_video)
# Test video files(provide a col name, since no col name is there)

test_videos = pd.DataFrame(list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER))), columns=['video'])

test_videos.head()
# show one of test

display_image_from_video_list(test_videos.sample(6).video, TEST_FOLDER)
# Face detection

class ObjectDetector():

    

    def __init__(self, object_cascade_path):

        self.objectCascade = cv.CascadeClassifier(object_cascade_path)

        

    def detect(self, image, scale_factor=1.3, min_neighbors=5, min_size=(20, 20)):

        rects = self.objectCascade.detectMultiScale(image, 

                                                    scaleFactor=scale_factor,

                                                    minNeighbors=min_neighbors,

                                                    minSize=min_size)

        return rects
frontal_cascade_path = os.path.join(FACE_DETECTION_FOLDER, 'haarcascade_frontalface_default.xml')

eye_cascade_path= os.path.join(FACE_DETECTION_FOLDER,'haarcascade_eye.xml')

profile_cascade_path= os.path.join(FACE_DETECTION_FOLDER,'haarcascade_profileface.xml')

smile_cascade_path= os.path.join(FACE_DETECTION_FOLDER,'haarcascade_smile.xml')



fd = ObjectDetector(frontal_cascade_path)

ed = ObjectDetector(eye_cascade_path)

pd = ObjectDetector(profile_cascade_path)

sd = ObjectDetector(smile_cascade_path)
def detect_objects(image, scale_factor, min_neighbors, min_size):

    

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    

    eyes = ed.detect(image_gray, scale_factor=scale_factor, min_neighbors=min_neighbors, min_size=(int(min_size[0]/2), int(min_size[1]/2)))

    

    for x, y, w, h in eyes:

        cv.circle(image, (int(x+w/2), int(y+h/2)), (int((w + h)/4)), (0, 0, 255), 3)

    

    profiles = pd.detect(image_gray, scale_factor=scale_factor, min_neighbors=min_neighbors, min_size=min_size)

    

    for x, y, w, h in profiles:

        cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)

    

    faces = fd.detect(image_gray, scale_factor=scale_factor, min_neighbors=min_neighbors, min_size=min_size)



    for x, y, w, h in faces:

        cv.rectangle(image,(x, y),(x+w, y+h),(0, 255, 0), 3)

    

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    ax.imshow(image)
def extract_image_objects(video_file, video_set_folder=TRAIN_SAMPLE_FOLDER):

    

    video_path = os.path.join(DATA_FOLDER, video_set_folder, video_file)

    cap_image = cv.VideoCapture(video_path)

    ret, frame = cap_image.read()

    

    detect_objects(image=frame, scale_factor=1.3, min_neighbors=5, min_size=(50, 50))
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='kgbkktcjxf.mp4'].index)

for video_file in same_original_fake_train_sample_video[1:4]:

    print(video_file)

    extract_image_objects(video_file)
fake_videos = list(meta_train_df.loc[meta_train_df.label == 'FAKE'].index)
from IPython.display import HTML

from base64 import b64encode



def play_video(video_file, subset=TRAIN_SAMPLE_FOLDER):

    video_url = open(os.path.join(DATA_FOLDER, subset, video_file),'rb').read()

    data_url = "data:video/mp4;base64," + b64encode(video_url).decode()

    return HTML("""<video width=500 controls><source src="%s" type="video/mp4"></video>""" % data_url)
play_video(fake_videos[0])
!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.0.0-py3-none-any.whl



from facenet_pytorch.models.inception_resnet_v1 import get_torch_home

torch_home = get_torch_home()



# Copy model checkpoints to torch cache so they are loaded automatically by the package

!mkdir -p $torch_home/checkpoints/

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth $torch_home/checkpoints/vggface2_DG3kwML46X.pt

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth $torch_home/checkpoints/vggface2_G5aNV2VSMn.pt
import os

import glob

import time

import torch

import cv2

from PIL import Image

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm



# See github.com/timesler/facenet-pytorch:

from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face



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
bias = -0.2942

weight = 0.68235746



submission = []

for filename, x_i in zip(filenames, X):

    if x_i is not None:

        prob = 1 / (1 + np.exp(-(bias + (weight * x_i).mean())))

    else:

        prob = 0.5

    submission.append([os.path.basename(filename), prob])
submission = pd.DataFrame(submission, columns=['filename', 'label'])

submission.sort_values('filename').to_csv('submission.csv', index=False)



plt.hist(submission.label, 20)

plt.show()