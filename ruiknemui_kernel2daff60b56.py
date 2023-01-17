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
# https://www.kaggle.com/timesler/comparison-of-face-detection-packages　を参考にした
%%capture

!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.0.0-py3-none-any.whl
import cv2

from matplotlib import pyplot as plt

from PIL import Image

import torch

from tqdm.notebook import tqdm

import time



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

device
sample = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/aagfhgtpmv.mp4'



reader = cv2.VideoCapture(sample)

images_1080_1920 = []

images_720_1280 = []

images_540_960 = []
print(reader.get(cv2.CAP_PROP_FRAME_WIDTH))

print(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(reader.get(cv2.CAP_PROP_FRAME_COUNT))
for i in tqdm(range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT)))):

    _, image = reader.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images_1080_1920.append(image)

    images_720_1280.append(cv2.resize(image, (1280, 720)))

    images_540_960.append(cv2.resize(image, (960, 540)))
reader.release()
images_1080_1920 = np.stack(images_1080_1920)

images_720_1280 = np.stack(images_720_1280)

images_540_960 = np.stack(images_540_960)



print('Shapes:')

print(images_1080_1920.shape)

print(images_720_1280.shape)

print(images_540_960.shape)
def plot_faces(images, figsize=(10.8/2, 19.2/2), start_frame=0, end_frame=0):

    shape = images[0].shape

    if end_frame == 0:

        end_frame = len(images) - 1

    images = images[np.linspace(start_frame, end_frame, 16).astype(int)]

    im_plot = []

    for i in range(0, 16, 4):

        im_plot.append(np.concatenate(images[i:i+4], axis=1))

    im_plot = np.concatenate(im_plot, axis=0)

    

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.imshow(im_plot)

    ax.xaxis.set_visible(False)

    ax.yaxis.set_visible(False)



    ax.grid(False)

    fig.tight_layout()



def timer(detector, detect_fn, images, *args):

    start = time.time()

    faces = detect_fn(detector, images, *args)

    elapsed = time.time() - start

    print(f', {elapsed:.3f} seconds')

    return faces, elapsed
plot_faces(images_1080_1920, figsize=(10.8, 19.2), start_frame=0, end_frame=15)
from facenet_pytorch import MTCNN

detector = MTCNN(device=device, post_process=False)



def detect_facenet_pytorch(detector, images, batch_size):

    faces = []

    for lb in np.arange(0, len(images), batch_size):

        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]

        faces.extend(detector(imgs_pil))

    return faces



times_facenet_pytorch = []    # batched
print('Detecting faces in 1080x1920 frames', end='')

faces, elapsed = timer(detector, detect_facenet_pytorch, images_1080_1920, 20)

times_facenet_pytorch.append(elapsed)



plot_faces(torch.stack(faces).permute(0, 2, 3, 1).int().numpy())
torch.stack(faces).shape
plot_faces(torch.stack(faces).permute(0, 3, 2, 1).int().numpy())
plot_faces(torch.stack(faces).permute(0, 2, 3, 1).int().numpy(), start_frame=0, end_frame=15)
from IPython.display import HTML

from base64 import b64encode



DATA_FOLDER = '../input/deepfake-detection-challenge'

TRAIN_SAMPLE_FOLDER = 'train_sample_videos'



def play_video(video_file, subset=TRAIN_SAMPLE_FOLDER):

    '''

    Display video

    param: video_file - the name of the video file to display

    param: subset - the folder where the video file is located (can be TRAIN_SAMPLE_FOLDER or TEST_Folder)

    '''

    video_url = open(os.path.join(DATA_FOLDER, subset,video_file),'rb').read()

    data_url = "data:video/mp4;base64," + b64encode(video_url).decode()

    return HTML("""<video width=500 controls><source src="%s" type="video/mp4"></video>""" % data_url)
play_video("aagfhgtpmv.mp4")
plot_faces(torch.stack(faces).permute(0, 2, 3, 1).int().numpy(), start_frame=16, end_frame=31)
plot_faces(torch.stack(faces).permute(0, 2, 3, 1).int().numpy(), start_frame=32, end_frame=48)
faces2 = torch.stack(faces).permute(0, 2, 3, 1).int().numpy()
plt.imshow(faces2[0])
cv2.cvtColor(faces2[0], cv2.COLOR_RGB2GRAY)
faces2[0].dtype
tmp_gray_face = cv2.cvtColor(faces2[0].astype("uint16"), cv2.COLOR_RGB2GRAY)

plt.imshow(tmp_gray_face)
plt.imshow(tmp_gray_face, cmap = "gray")

# https://tetlab117.hatenablog.com/entry/2017/09/28/163638

bf = cv2.BFMatcher(cv2.NORM_HAMMING)



# ORBとAKAZEは特徴点や特徴量を抽出するアルゴリズム

# コメントアウトを調節することによりどちらでも行える



# detector = cv2.ORB_create()

detector = cv2.AKAZE_create()
# まずは0フレーム目と1フレーム目の類似度

target_img = cv2.cvtColor(faces2[0].astype("uint16"), cv2.COLOR_RGB2GRAY)

comparing_img = cv2.cvtColor(faces2[1].astype("uint16"), cv2.COLOR_RGB2GRAY)
fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1)

ax1.imshow(target_img, cmap = "gray")

ax2 = fig.add_subplot(1, 2, 2)

ax2.imshow(comparing_img, cmap = "gray")

plt.show()
detector = cv2.AKAZE_create()

(target_kp, target_des) = detector.detectAndCompute(target_img, None)

target_kp
target_img = faces2[0].astype("uint16")

comparing_img = faces2[1].astype("uint16")
target_hist = cv2.calcHist([target_img], [0], None, [256], [0, 256])

comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])

ret = cv2.compareHist(target_hist, comparing_hist, 0)

print(ret)
hist_similarity = []

for i in range(0,299,1):

    target_hist = cv2.calcHist([faces2[i].astype("uint16")], [0], None, [256], [0, 256])

    comparing_hist = cv2.calcHist([faces2[i+1].astype("uint16")], [0], None, [256], [0, 256])

    hist_similarity.append(cv2.compareHist(target_hist, comparing_hist, 0))
plt.figure(figsize=(20, 4))

plt.plot(hist_similarity)
plot_faces(torch.stack(faces).permute(0, 2, 3, 1).int().numpy(), start_frame=125, end_frame=140)
def plot_histgram(file):

    sample = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/' + file

    images_1080_1920 = []



    reader = cv2.VideoCapture(sample)



    for i in tqdm(range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT)))):

        _, image = reader.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images_1080_1920.append(image)

    reader.release()



    images_1080_1920 = np.stack(images_1080_1920)



    detector = MTCNN(device=device, post_process=False)

    faces, elapsed = timer(detector, detect_facenet_pytorch, images_1080_1920, 20)

    times_facenet_pytorch.append(elapsed)



    faces2 = torch.stack(faces).permute(0, 2, 3, 1).int().numpy()





    hist_similarity = []

    for i in range(0,100,1):

        target_hist = cv2.calcHist([faces2[i].astype("uint16")], [0], None, [256], [0, 256])

        comparing_hist = cv2.calcHist([faces2[i+1].astype("uint16")], [0], None, [256], [0, 256])

        hist_similarity.append(cv2.compareHist(target_hist, comparing_hist, 0))



    plt.figure(figsize=(20, 4))

    plt.plot(hist_similarity)
DATA_FOLDER = '../input/deepfake-detection-challenge'

TRAIN_SAMPLE_FOLDER = 'train_sample_videos'



def get_meta_from_json(path):

    df = pd.read_json(os.path.join(DATA_FOLDER, path, "metadata.json"))

    df = df.T

    return df



meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER)

meta_train_df.head()
for file in meta_train_df.loc[meta_train_df.label == "FAKE"].head(10).index.to_list():

    plot_histgram(file)
canny_img = cv2.Canny(cv2.cvtColor(faces2[0].astype("uint8"), cv2.COLOR_RGB2GRAY), 50, 110)

# Cannyはuint16はだめらしい
plt.imshow(canny_img, cmap = "gray")
canny_img = cv2.Canny(cv2.cvtColor(faces2[45].astype("uint8"), cv2.COLOR_RGB2GRAY), 50, 110)

plt.imshow(canny_img, cmap = "gray")