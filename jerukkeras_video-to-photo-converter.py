#import library

import cv2

import os

import pandas as pd
#cek input data

os.listdir("../input/videodata-for-hog")

#membuat folder video

os.mkdir("video")
vid = pd.DataFrame(os.listdir("../input/videodata-for-hog"))

vid.tail
path = "../input/videodata-for-hog"

vids = os.listdir(path)



for v in range (len(vids)):

    vid = os.path.join(path, vids[v])

    vid = cv2.VideoCapture(vid)

    

    i = 0

    while True:

        ret, frame = vid.read()

        if not ret:

            break

        cv2.imwrite("video/" + str(v) + "-" + str(i) + ".jpg", frame)

        i = i + 1

vid = pd.DataFrame(os.listdir("video/"))

vid.tail()

from IPython.display import Image

#melihat hasil gambar

Image(filename='video/0-1.jpg')