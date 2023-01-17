#!pip install dlib

!pip install face_recognition

#!pip install pillow

import numpy as np # linear algebra

import pandas as pd # data processing

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

face_locations=face_recognition.face_locations(image)

num_of_faces=len(face_locations)

print("{} face(s) found in this image".format(num_of_faces))



pil_image=PIL.Image.fromarray(image)



print("Pixel locations of faces: ")



#face locations are defined and rectange is drawn

for face_location in face_locations:

    top,left,bottom,right=face_location

    print("A face is located at pixel location at Top: {} Left: {} Bottom: {} Right: {}".format(top,left,bottom,right))

    

    draw=PIL.ImageDraw.Draw(pil_image)

    draw.rectangle([left,top,right,bottom],outline="blue",width=3)

    

pil_image.show()

plt.imshow(pil_image)

plt.show()