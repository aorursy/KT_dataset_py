import numpy as np

import pandas as pd

import os

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

%matplotlib inline 

import cv2 as cv
FACE_DETECTION_FOLDER = "..//input//cat-face-detection//"

#Frontal cat face detector

frontal_cascade_path = os.path.join(FACE_DETECTION_FOLDER,'haarcascade_frontalcatface.xml')
class CatFaceDetector():

    '''

    Class for Cat Face Detection

    '''

    def __init__(self,object_cascade_path):

        '''

        param: object_cascade_path - path for the *.xml defining the parameters for cat face detection algorithm

        source of the haarcascade resource is: https://github.com/opencv/opencv/tree/master/data/haarcascades

        '''



        self.objectCascade=cv.CascadeClassifier(object_cascade_path)





    def detect(self, image, scale_factor=1.15,

               min_neighbors=1,

               min_size=(30,30)):

        '''

        Function return rectangle coordinates of cat face for given image

        param: image - image to process

        param: scale_factor - scale factor used for cat face detection

        param: min_neighbors - minimum number of parameters considered during cat face detection

        param: min_size - minimum size of bounding box for object detected

        '''

        bbox=self.objectCascade.detectMultiScale(image,

                                                scaleFactor=scale_factor,

                                                minNeighbors=min_neighbors,

                                                minSize=min_size)

        return bbox
#Detector for cat frontal face detectiob created

fcfd=CatFaceDetector(frontal_cascade_path)
def detect_cat_face(image, scale_factor, min_neighbors, min_size):

    '''

    Cat Face detection function

    Identify frontal cat face and display the detected marker over the image

    param: image - the image extracted from the video

    param: scale_factor - scale factor parameter for `detect` function of ObjectDetector object

    param: min_neighbors - min neighbors parameter for `detect` function of ObjectDetector object

    param: min_size - minimum size parameter for f`detect` function of ObjectDetector object

    '''

    

    image_gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)



    cat_face=fcfd.detect(image_gray,

                   scale_factor=scale_factor,

                   min_neighbors=min_neighbors,

                   min_size=min_size)



    for x, y, w, h in cat_face:

        #detected cat face shown in color image

        cv.circle(image,(int(x+w/2),int(y+h/2)),(int((w + h)/4)),(0, 127,255),3)



    # image

    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(111)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    ax.imshow(image)

    #return image
os.listdir("..//input//cat-dataset//CAT_00")



file_list = []

for root, dirs, files in os.walk("..//input//cat-dataset//CAT_00"):

    for file in files:

        if file.endswith(".jpg"):    

            file_list.append(file)
excerpt_file_list = file_list[0:10]

excerpt_file_list
# cat image

for img_src in excerpt_file_list:

    img_source = cv.imread(os.path.join("..//input//cat-dataset//CAT_00",img_src))

    # detect face and show cat face marker over image

    detect_cat_face(image=img_source,scale_factor=1.10, min_neighbors=3, min_size=(50, 50)) 