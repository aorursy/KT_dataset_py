import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

from skimage.feature import hog

from skimage import data, exposure

import sys

import cv2

import matplotlib

from subprocess import check_output



from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

#list the files in the input directory

#print(os.listdir("../input"))

#print(check_output(["ls", "../input"]).decode("utf8")) #trainLabels.csv

#print(check_output(["pwd", ""]).decode("utf8")) # /kaggle/working/

#classes : 0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, 4 - Proliferative DR

def classes_to_int(label):

    # label = classes.index(dir)

    label = label.strip()

    if label == "No DR":  return 0

    if label == "Mild":  return 1

    if label == "Moderate":  return 2

    if label == "Severe":  return 3

    if label == "Proliferative DR":  return 4

    print("Invalid Label", label)

    return 5



def int_to_classes(i):

    if i == 0: return "No DR"

    elif i == 1: return "Mild"

    elif i == 2: return "Moderate"

    elif i == 3: return "Severe"

    elif i == 4: return "Proliferative DR"

    print("Invalid class ", i)

    return "Invalid Class"
NUM_CLASSES = 5

# we need images of same size so we convert them into the size

WIDTH = 128

HEIGHT = 128

DEPTH = 3

inputShape = (HEIGHT, WIDTH, DEPTH)

# initialize number of epochs to train for, initial learning rate and batch size

EPOCHS = 15

INIT_LR = 1e-3

BS = 32

#global variables

ImageNameDataHash = {}

uniquePatientIDList = []
def readTrainData(trainDir):

    global ImageNameDataHash

    # loop over the input images

    images = os.listdir(trainDir)

    print("Number of files in " + trainDir + " is " + str(len(images)))

    for imageFileName in images:

        if (imageFileName == "trainLabels.csv"):

            continue

        # load the image, pre-process it, and store it in the data list

        imageFullPath = os.path.join(os.path.sep, trainDir, imageFileName)

        #print(imageFullPath)

        

        image = load_img(imageFullPath)

        

        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),

                    cells_per_block=(1, 1), visualize=True, multichannel=True)



        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)



        ax1.axis('off')

        ax1.imshow(image, cmap=plt.cm.gray)

        ax1.set_title('Input image')



# Rescale histogram for better display

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))



        ax2.axis('off')

        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)

        ax2.set_title('Histogram of Oriented Gradients')

        plt.show()

    return 

        


from datetime import datetime

print("Loading images at..."+ str(datetime.now()))

sys.stdout.flush()

readTrainData("/kaggle/working/../input/")

print("Loaded " + str(len(ImageNameDataHash)) + " images at..."+ str(datetime.now())) # 1000