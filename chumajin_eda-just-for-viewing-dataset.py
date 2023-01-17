import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



allimagepath = []

testimagepath = []

allindex = []



TRAINCSV = "/kaggle/input/landmark-retrieval-2020/train.csv"



for dirname, _, filenames in os.walk('/kaggle/input/landmark-retrieval-2020/train'):

    for filename in filenames:

        allimagepath.append(os.path.join(dirname, filename))



for dirname, _, filenames in os.walk('/kaggle/input/landmark-retrieval-2020/test'):

    for filename in filenames:

        testimagepath.append(os.path.join(dirname, filename))





for dirname, _, filenames in os.walk('/kaggle/input/landmark-retrieval-2020/index'):

    for filename in filenames:

        allindex.append(os.path.join(dirname, filename))
import cv2

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/landmark-retrieval-2020/train.csv")

df
landmark_idlist = df["landmark_id"].drop_duplicates()

landmark_idlist = landmark_idlist.reset_index(drop=True)

landmark_idlist
def imageshow(id):

    tmpdf = df[df["landmark_id"]==id].reset_index(drop=True)

    idlist = tmpdf["id"]

    

    for a in idlist:

        impath = [s for s in allimagepath if a in s]

        img = cv2.imread(impath[0])

        

        plt.figure()

        plt.imshow(img)

        

print("landmark_id = " + str(landmark_idlist[0]))

imageshow(landmark_idlist[0])
print("landmark_id = " + str(landmark_idlist[200]))

imageshow(landmark_idlist[200])
allindex[0]
for a in range(20):

    img = cv2.imread(allindex[a])

    plt.figure()

    plt.imshow(img)
testimagepath[0]
for a in range(20):

    img = cv2.imread(testimagepath[a])

    plt.figure()

    plt.imshow(img)