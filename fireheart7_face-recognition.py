import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import tensorflow as tf

from tqdm import tqdm_notebook as tqdm



%matplotlib inline
for dirname, _, _ in os.walk("/kaggle/input") : 

    print(dirname)
image_paths = []

image_names = []

image_dir = "/kaggle/input/ffhq-face-data-set/thumbnails128x128/"

for image_name in tqdm(os.listdir(image_dir)) : 

    image_path = image_dir + image_name

    image_paths.append(image_path)

    image_names.append(image_name)    
len(image_names), len(image_paths)
image_dataframe = pd.DataFrame(index = np.arange(len(image_names)), columns = ["image_name", "path"])



i = 0 

for name, path in tqdm(zip(image_names, image_paths)) : 

    image_dataframe.iloc[i]["image_name"] = name

    image_dataframe.iloc[i]["path"] = path

    i = i + 1



print("Dataframe shape = ", image_dataframe.shape)
image_dataframe.head()
sample_images = []
def get_images() : 

    sample_images = []

    random_image_paths = [np.random.choice(image_dataframe["path"]) for i in range(6)]



    plt.figure(figsize = (12, 8))

    for i in range(6) : 

        plt.subplot(2,3, i+1)

        image = cv2.imread(random_image_paths[i])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sample_images.append(image)

        plt.imshow(image, cmap = "gray")

        plt.grid(False)

    plt.tight_layout() # Automatically adjust subplot parameters to give specified padding.

    return sample_images
sample_images = get_images()
def haar_cascade_detection(sample_images) : 

    face_cascade = cv2.CascadeClassifier("../input/haar-cascades-for-face-detection/haarcascade_frontalface_default.xml")



    for image in tqdm(sample_images) : 

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.05, 5, 50)

    

        for (x_coordinate, y_coordinate, height, width) in faces : 

            cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width, y_coordinate + height), (100, 0, 0), 2)
haar_cascade_detection(sample_images)
plt.figure(figsize = (12, 8))

for i in range(6) : 

    plt.subplot(2,3, i+1)

    plt.imshow(sample_images[i], cmap = "gray")

    plt.title("Image {}".format(i+1))

    plt.grid(False)

plt.tight_layout() # Automatically adjust subplot parameters to give specified padding.
def load_my_image() : 

    my_image_01 = cv2.imread("../input/myimages/img_01.jpg")

    my_image_01 = cv2.cvtColor(my_image_01, cv2.COLOR_BGR2RGB)



    my_image_02 = cv2.imread("../input/myimages/img_2.jpg")

    my_image_02 = cv2.cvtColor(my_image_02, cv2.COLOR_BGR2RGB)



    plt.figure(figsize = (12, 8))

    plt.subplot(1,2,1) 

    plt.title("The One During A Conference", fontsize = 13)

    plt.imshow(my_image_01, cmap = "gray")

    plt.grid(False)



    plt.subplot(1,2,2) 

    plt.title("The One With Home Buddies!", fontsize = 13)

    plt.imshow(my_image_02, cmap = "gray")

    plt.grid(False)



    plt.tight_layout()

    return my_image_01, my_image_02
my_image_01, my_image_02 = load_my_image()
haar_cascade_detection([my_image_01, my_image_02])
plt.figure(figsize = (12, 8))

plt.subplot(1,2,1) 

plt.title("The One During A Conference", fontsize = 13)

plt.imshow(my_image_01, cmap = "gray")

plt.grid(False)



plt.subplot(1,2,2) 

plt.title("The One With Home Buddies!", fontsize = 13)

plt.imshow(my_image_02, cmap = "gray")

plt.grid(False)



plt.tight_layout()
! pip install mtcnn
sample_images = get_images()
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
def mtcnn_detector(sample_images) : 

    for image in tqdm(sample_images) : 

        face_location = detector.detect_faces(image)

        for face in zip(face_location) : 

            x_coordinate, y_coordinate, width, height = face[0]['box']

            #face_landmarks = face[0]['keypoints']

            cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width, y_coordinate + height), (0,0,100), 2)
mtcnn_detector(sample_images)
plt.figure(figsize = (12, 8))

for i in range(6) : 

    plt.subplot(2,3, i+1)

    plt.imshow(sample_images[i], cmap = "gray")

    plt.title("Image {}".format(i+1))

    plt.grid(False)

plt.tight_layout() # Automatically adjust subplot parameters to give specified padding.
my_image_01, my_image_02 = load_my_image()
mtcnn_detector([my_image_01, my_image_02])
plt.figure(figsize = (12, 8))

plt.subplot(1,2,1) 

plt.title("The One During A Conference", fontsize = 13)

plt.imshow(my_image_01, cmap = "gray")

plt.grid(False)



plt.subplot(1,2,2) 

plt.title("The One With Home Buddies!", fontsize = 13)

plt.imshow(my_image_02, cmap = "gray")

plt.grid(False)



plt.tight_layout()