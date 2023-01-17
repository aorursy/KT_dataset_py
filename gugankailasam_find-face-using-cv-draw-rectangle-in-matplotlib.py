import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
def label_transform(age):
    age = tf.cast(age, dtype='float32')
    return age


from torchvision.datasets import ImageFolder
train_data = ImageFolder(root='/kaggle/input/age-prediction/age_prediction_up/age_prediction/train/')
test_data = ImageFolder(root='/kaggle/input/age-prediction//age_prediction_up/age_prediction/test/')
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def find_face(x):

    # Read image from your local file system
    original_image = cv.imread(x)

    # Convert color image to grayscale for Viola-Jones
    grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    # Load the classifier and create a cascade object for face detection
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    detected_faces = face_cascade.detectMultiScale(grayscale_image)
    
    if detected_faces != ():
        
        column, row, width, height = detected_faces.ravel()[0:4]

        # Create figure and axes
        fig,ax = plt.subplots(1)

        # Display the image
        ax.imshow(plt.imread(x))

        # Create a Rectangle patch
        rect = patches.Rectangle((column,row),width, height,
                                 linewidth=4, edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
start = int(14*1e4)
for img in train_data.imgs[start:start+5]:
    find_face(img[0])
    plt.show()