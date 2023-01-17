# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install dlib
!pip install face_recognition
!pip install imutils
from imutils import paths

import face_recognition

import argparse

import pickle

import cv2

import os

imagePaths = list(paths.list_images("/kaggle/input/sdcfsdsd/x/bn/"))
knownEncodings = []

knownNames = []

for (i, imagePath) in enumerate(imagePaths):

    # extract the person name from the image path

    print("[INFO] processing image {}/{}".format(i + 1,

        len(imagePaths)))

    name = imagePath.split(os.path.sep)[-2]



    # load the input image and convert it from RGB (OpenCV ordering)

    # to dlib ordering (RGB)

    image = cv2.imread(imagePath)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    # detect the (x, y)-coordinates of the bounding boxes

    # corresponding to each face in the input image

    boxes = face_recognition.face_locations(rgb,

        model=["cnn"])



    # compute the facial embedding for the face

    encodings = face_recognition.face_encodings(rgb, boxes)



    # loop over the encodings

    for encoding in encodings:

        # add each encoding + name to our set of known names and

        # encodings

        knownEncodings.append(encoding)

        knownNames.append(name)
data = {"encodings": knownEncodings, "names": knownNames}
f = open("/kaggle/working/encodings.pickle", "wb")

f.write(pickle.dumps(data))

f.close()
print("[INFO] loading encodings...")

model = pickle.loads(open("/kaggle/working/encodings.pickle", "rb").read())
image = cv2.imread("/kaggle/input/sdcfsdsd/x/Screenshot from 2020-03-02 16-12-49.png")#test-image/test/29bolly6.jpg")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("[INFO] recognizing faces...")

boxes = face_recognition.face_locations(rgb,

    model=["cnn"])

encodings = face_recognition.face_encodings(rgb, boxes)
# initialize the list of names for each face detected

names = []
# loop over the facial embeddings

for encoding in encodings:

    # attempt to match each face in the input image to our known

    # encodings

    matches = face_recognition.compare_faces(model["encodings"],

        encoding)

    name = "Unknown"



    # check to see if we have found a match

    if True in matches:

        # find the indexes of all matched faces then initialize a

        # dictionary to count the total number of times each face

        # was matched

        matchedIdxs = [i for (i, b) in enumerate(matches) if b]

        counts = {}



        # loop over the matched indexes and maintain a count for

        # each recognized face face

        for i in matchedIdxs:

            name = model["names"][i]

            counts[name] = counts.get(name, 0) + 1



        # determine the recognized face with the largest number of

        # votes (note: in the event of an unlikely tie Python will

        # select first entry in the dictionary)

        name = max(counts, key=counts.get)

    

    # update the list of names

    names.append(name)

# loop over the recognized faces

for ((top, right, bottom, left), name) in zip(boxes, names):

    # draw the predicted face name on the image

    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    y = top - 15 if top - 15 > 15 else top + 15

    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,

        0.75, (0, 255, 0), 2)
ds=image
from matplotlib import pyplot as plt
name
print(plt.imshow(image),name)