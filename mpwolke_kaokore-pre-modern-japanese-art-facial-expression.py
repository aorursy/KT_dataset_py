#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQOCSum1o4vBFAGpLPXrFM8HLeQ4LcJfqPbGqSgVkDb3MMC1ndj',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQXRagmVaN3mEo_D2na2YDmaWpsQzcyIbmUR3pvPOqUCsrxQbA3',width=400,height=400)
from fastai import *

from fastai.vision import *

from tqdm import tqdm_notebook as tqdm

import os

import cv2



import random

import numpy as np

import keras

from random import shuffle

from keras.utils import np_utils

from shutil import unpack_archive

import matplotlib.pyplot as plt

import math

import os

import tensorflow as tf



%reload_ext autoreload

%autoreload 2

%matplotlib inline
!pip install imutils
"""#Preprocess the images such that only the hand sign goes through using opencv



import imutils



# global variables

bg = None



def run_avg(image, aWeight):

    global bg

    # initialize the background

    if bg is None:

        bg = image.copy().astype("float")

        return



    # compute weighted average, accumulate it and update the background

    cv2.accumulateWeighted(image, bg, aWeight)







def segment(image, threshold=25):

    global bg

    # find the absolute difference between background and current frame

    diff = cv2.absdiff(bg.astype("uint8"), image)



    # threshold the diff image so that we get the foreground

    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]



    # get the contours in the thresholded image

    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    # return None, if no contours detected

    if len(cnts) == 0:

        return

    else:

        # based on contour area, get the maximum contour which is the hand

        segmented = max(cnts, key=cv2.contourArea)

        return (thresholded, segmented)







if __name__ == "__main__":

    # initialize weight for running average

    aWeight = 0.5



    # get the reference to the webcam

    camera = cv2.VideoCapture(video_path) 



    # region of interest (ROI) coordinates

    top, right, bottom, left = 10, 350, 225, 590



    # initialize num of frames

    num_frames = 0



    # keep looping, until interrupted

    while(True):

        # get the current frame

        (grabbed, frame) = camera.read()



        # resize the frame

        frame = imutils.resize(frame, width=700)



        # flip the frame so that it is not the mirror view

        frame = cv2.flip(frame, 1)



        # clone the frame

        clone = frame.copy()        

        

        

        # get the height and width of the frame

        (height, width) = frame.shape[:2]



        # get the ROI

        roi = frame[top:bottom, right:left]



        # convert the roi to grayscale and blur it

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (7, 7), 0)



        # to get the background, keep looking till a threshold is reached

        # so that our running average model gets calibrated

        if num_frames < 30:

            run_avg(gray, aWeight)

        else:

            # segment the hand region

            hand = segment(gray)



            # check whether hand region is segmented

            if hand is not None:

                # if yes, unpack the thresholded image and

                # segmented region

                (thresholded, segmented) = hand



                # draw the segmented region and display the frame

                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                cv2.imshow("Thesholded", thresholded)

                

                

                

        # draw the segmented hand

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)



        # increment the number of frames

        num_frames += 1



        # display the frame with segmented hand

        cv2.imwrite(filename="/kaggle/working/screens/alpha.png", img=clone)



        # observe the keypress by the user

        keypress = cv2.waitKey(1) & 0xFF



        # if the user pressed "q", then stop looping

        if keypress == ord("q"):

            break





cv2.destroyAllWindows()



"""
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQIyDeJ5FapwfXnqTDd1hPRsrI8XM2MJjAEjpSDMLbGU-JbltSw',width=400,height=400)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/kaokore/images/fig/label_example.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
femalewarrior_file = '../input/kaokore/dataset/labels.metadata.ja.txt'

with open(femalewarrior_file) as f: # The with keyword automatically closes the file when you are done

    print (f.read(1000))
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQe51pdk4XCBNv-BLzKFKDotfjmd6sk36-VyCWzQhwEQvsrSreq',width=400,height=400)
# When transforming the dataset make sure that the zoom does not exceed too much and the cropping does not crop the hand sign



data = (ImageList.from_folder("/kaggle/input/kaokore/images/fig/")

        .split_by_rand_pct()          

        .label_from_folder()

        .add_test_folder() 

        .transform(get_transforms(),size=224)

        .databunch()

        .normalize(imagenet_stats))



#data = (ImageDataBunch.from_folder(mainPath) .random_split_by_pct() .label_from_folder() .transform(tfms, size=224) .databunch())

#data = (ImageList.from_folder(mainPath) .split_by_rand_pct() .label_from_folder() .databunch())



data
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSLI3OqiF5R6rPCEQ4LB2bxxGrOc_ednDnams41naaOB6HbyRmN',width=400,height=400)
kaokore_file = '../input/kaokore/dataset/original_tags.txt'

with open(kaokore_file) as f: # The with keyword automatically closes the file when you are done

    print (f.read(1000))
import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

def plotWordFrequency(input):

    f = open(kaokore_file,'r')

    words = [x for y in [l.split() for l in f.readlines()] for x in y]

    data = sorted([(w, words.count(w)) for w in set(words)], key = lambda x:x[1], reverse=True)[:40] 

    most_words = [x[0] for x in data]

    times_used = [int(x[1]) for x in data]

    plt.figure(figsize=(20,10))

    plt.bar(x=sorted(most_words), height=times_used, color = 'grey', edgecolor = 'black',  width=.5)

    plt.xticks(rotation=45, fontsize=18)

    plt.yticks(rotation=0, fontsize=18)

    plt.xlabel('Most Common Words:', fontsize=18)

    plt.ylabel('Number of Occurences:', fontsize=18)

    plt.title('Most Commonly Used Words: %s' % (kaokore_file), fontsize=24)

    plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
kaokore_file = '../input/kaokore/dataset/original_tags.txt'

plotWordFrequency(kaokore_file)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/kaokore/images/fig/dataset_example.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSR3mB4lh4iBchbUFGI7uCD9LR0nTIxLFmGHFEffUd77N1_HqfF',width=400,height=400)