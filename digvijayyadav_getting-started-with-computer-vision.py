!pip install mtcnn
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import cv2 as cv2

from cv2 import CascadeClassifier, rectangle

from cv2 import destroyAllWindows

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import *

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import pyplot

from matplotlib.patches import Rectangle, Circle

from mtcnn.mtcnn import MTCNN

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
images_dir = '../input/face-mask-detection/images/'
from IPython.display import YouTubeVideo



YouTubeVideo('w4tigQn-7Jw', width = 450)
filename = '../input/face-mask-detection/images/maksssksksss1.png'

pixels = cv2.imread(filename)



#creating a detector with default weights

detector = MTCNN()



faces = detector.detect_faces(pixels)



for face in faces:

    print(face)
def draw_image_with_boxes(filename, result_list):

    # load the image

    data = pyplot.imread(filename)

    # plot the image

    plt.imshow(data)

    # get the context for drawing boxes

    ax = pyplot.gca()

    # plot each box

    for result in result_list:

        # get coordinates

        x, y, width, height = result['box']

        # create the shape

        rect = Rectangle((x, y), width, height, fill=False, color='red')

        # draw the box

        ax.add_patch(rect)

        # draw the dots on eyes nose ..

        for key, value in result['keypoints'].items():

            # create and draw dot

            dot = Circle(value, radius=2, color='red')

            ax.add_patch(dot)

    # show the plot

    pyplot.show()
filename = '../input/face-mask-detection/images/maksssksksss10.png'

pixels = cv2.imread(filename)

# create the detector, using default weights

detector = MTCNN()

# detect faces in the image

faces = detector.detect_faces(pixels)

# display faces on the original image

draw_image_with_boxes(filename, faces)
filename = '../input/face-mask-detection/images/maksssksksss121.png'

pixels = cv2.imread(filename)

# create the detector, using default weights

detector = MTCNN()

# detect faces in the image

faces = detector.detect_faces(pixels)

# display faces on the original image

draw_image_with_boxes(filename, faces)
def draw_faces(filename, result_list):

    #loading the image

    data = cv2.imread(filename)

    

    # plot each face as subplot

    for i in range(len(result_list)):

        x1, y1, width, height = result_list[i]['box']

        x2, y2 = x1 + width, y1 + height

        

        #define subplot

        plt.subplot(1, len(result_list), i+1)

        plt.axis('off')

        

        plt.imshow(data[y1:y2, x1:x2])

    

    pyplot.show()



    

filename = '../input/face-mask-detection/images/maksssksksss121.png'

pixels = cv2.imread(filename)

# create the detector, using default weights

detector = MTCNN()

# detect faces in the image

faces = detector.detect_faces(pixels)



#Let's apply the defined function

draw_faces(filename, faces)