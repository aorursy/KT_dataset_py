#!pip install dlib

!pip install face_recognition

#!pip install pillow

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import PIL.Image

import PIL.ImageDraw

import face_recognition



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#data loading

image=face_recognition.load_image_file("../input/input-people/friends.jpg")

face_landmarks_list=face_recognition.face_landmarks(image)

num_of_faces=len(face_landmarks_list)

print("{} face(s) found in this image".format(num_of_faces))



pil_image=PIL.Image.fromarray(image)

draw=PIL.ImageDraw.Draw(pil_image)



for face_landmarks in face_landmarks_list:

    

    for name,list_of_points in face_landmarks.items():

        #location of the each facial feature

        print("The {} in this face has the following points: {}".format(name,list_of_points))

        

        draw.line(list_of_points, fill="blue", width=3)

        

pil_image.show()

plt.imshow(pil_image)

plt.show()

        