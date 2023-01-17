%%capture

!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.2.7-py3-none-any.whl

!pip install /kaggle/input/dlibpkg/dlib-19.19.0

!pip install /kaggle/input/mtcnn-package/mtcnn-0.1.0-py3-none-any.whl
import cv2

import numpy as np

from matplotlib import pyplot as plt

from PIL import Image

import torch

from tqdm.notebook import tqdm

import time



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sample = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/aagfhgtpmv.mp4'



reader = cv2.VideoCapture(sample)

images_1080_1920 = []

images_720_1280 = []

images_540_960 = []

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
def plot_faces(images, figsize=(10.8/2, 19.2/2)):

    shape = images[0].shape

    images = images[np.linspace(0, len(images)-1, 16).astype(int)]

    im_plot = []

    for i in range(0, 16, 4):

        im_plot.append(np.concatenate(images[i:i+4], axis=0))

    im_plot = np.concatenate(im_plot, axis=1)

    

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
plot_faces(images_540_960, figsize=(10.8, 19.2))
from facenet_pytorch import MTCNN

detector = MTCNN(device=device, post_process=False)



def detect_facenet_pytorch(detector, images, batch_size):

    faces = []

    for lb in np.arange(0, len(images), batch_size):

        imgs = [img for img in images[lb:lb+batch_size]]

        faces.extend(detector(imgs))

    return faces



times_facenet_pytorch = []    # batched

times_facenet_pytorch_nb = [] # non-batched
print('Detecting faces in 540x960 frames', end='')

_, elapsed = timer(detector, detect_facenet_pytorch, images_540_960, 60)

times_facenet_pytorch.append(elapsed)



print('Detecting faces in 720x1280 frames', end='')

_, elapsed = timer(detector, detect_facenet_pytorch, images_720_1280, 40)

times_facenet_pytorch.append(elapsed)



print('Detecting faces in 1080x1920 frames', end='')

faces, elapsed = timer(detector, detect_facenet_pytorch, images_1080_1920, 20)

times_facenet_pytorch.append(elapsed)



plot_faces(torch.stack(faces).permute(0, 2, 3, 1).int().numpy())
print('Detecting faces in 540x960 frames', end='')

_, elapsed = timer(detector, detect_facenet_pytorch, images_540_960, 1)

times_facenet_pytorch_nb.append(elapsed)



print('Detecting faces in 720x1280 frames', end='')

_, elapsed = timer(detector, detect_facenet_pytorch, images_720_1280, 1)

times_facenet_pytorch_nb.append(elapsed)



print('Detecting faces in 1080x1920 frames', end='')

faces, elapsed = timer(detector, detect_facenet_pytorch, images_1080_1920, 1)

times_facenet_pytorch_nb.append(elapsed)
del detector

torch.cuda.empty_cache()
from dlib import get_frontal_face_detector

detector = get_frontal_face_detector()



def detect_dlib(detector, images):

    faces = []

    for image in images:

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        boxes = detector(image_gray)

        box = boxes[0]

        face = image[box.top():box.bottom(), box.left():box.right()]

        faces.append(face)

    return faces



times_dlib = []
print('Detecting faces in 540x960 frames', end='')

_, elapsed = timer(detector, detect_dlib, images_540_960)

times_dlib.append(elapsed)



print('Detecting faces in 720x1280 frames', end='')

_, elapsed = timer(detector, detect_dlib, images_720_1280)

times_dlib.append(elapsed)



print('Detecting faces in 1080x1920 frames', end='')

faces, elapsed = timer(detector, detect_dlib, images_1080_1920)

times_dlib.append(elapsed)



plot_faces(np.stack([cv2.resize(f, (160, 160)) for f in faces]))
del detector

torch.cuda.empty_cache()
from mtcnn import MTCNN

detector = MTCNN()



def detect_mtcnn(detector, images):

    faces = []

    for image in images:

        boxes = detector.detect_faces(image)

        box = boxes[0]['box']

        face = image[box[1]:box[3]+box[1], box[0]:box[2]+box[0]]

        faces.append(face)

    return faces



times_mtcnn = []
print('Detecting faces in 540x960 frames', end='')

_, elapsed = timer(detector, detect_mtcnn, images_540_960)

times_mtcnn.append(elapsed)



print('Detecting faces in 720x1280 frames', end='')

_, elapsed = timer(detector, detect_mtcnn, images_720_1280)

times_mtcnn.append(elapsed)



print('Detecting faces in 1080x1920 frames', end='')

faces, elapsed = timer(detector, detect_mtcnn, images_1080_1920)

times_mtcnn.append(elapsed)



plot_faces(np.stack([cv2.resize(face, (160, 160)) for face in faces]))
del detector

torch.cuda.empty_cache()
fig, ax = plt.subplots(figsize=(10,6))



pos = np.arange(3)

plt.bar(pos, times_facenet_pytorch, 0.2, label='facenet-pytorch')

plt.bar(pos + 0.2, times_facenet_pytorch_nb, 0.2, label='facenet-pytorch (non-batched)')

plt.bar(pos + 0.4, times_dlib, 0.2, label='dlib')

plt.bar(pos + 0.6, times_mtcnn, 0.2, label='mtcnn')



ax.set_ylabel('Elapsed time (seconds)')

ax.set_xticks(pos + 0.25)

ax.set_xticklabels(['540x960', '720x1280', '1080x1920'])

plt.legend();