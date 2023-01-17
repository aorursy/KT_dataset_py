# Flowers Dataset

# Data Preparation

# This scripts reads the images from the different folders and assign a label
# the resulting files are:

# flowers.npz = images
# labels.npz = labels (daisy, dandelion, rose, sunflower, tulip)

# all images are resized to 160 x 160

# to load the data please use:

# << copy this code from here"
# Load npz file containing image arrays
# x_npz = np.load("../input/x_images_arrays.npz")
# x = x_npz['arr_0']
# Load binary encoded labels for Lung Infiltrations: 0=Not_infiltration 1=Infiltration
# y_npz = np.load("../input/y_infiltration_labels.npz")
# y = y_npz['arr_0']
# >> until here

# Import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile

import cv2
import os
import random
import matplotlib.pylab as plt
from glob import glob


from sklearn.model_selection import train_test_split

print("Flower Dataset Data Preparation")
print("Dr. Jose Mendoza")
print("06/22/2018")

# First, look where the flowers.zip file is.
print("Reading images from the following directories:")
from subprocess import check_output
print(check_output(["ls", "../input/flowers/flowers"]).decode("utf8"))
# Prepare directories and load images

labels = []
flowers = []
total = 0
# ../input/
PATH = os.path.abspath(os.path.join('..', 'input', 'flowers', 'flowers'))

# importing daisies

print("Importing images of daisy flowers")
# ../input/sample/images/
SOURCE_IMAGES = os.path.join(PATH, "daisy")

# ../input/sample/images/*.jpg
daisy = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

# Prepare labels for daisy

i = 0
while i < len(daisy):
    labels.append("daisy")
    flowers.append(daisy[i])
    i += 1
    total += 1

print("Imported "+ str(i) + " images of daisy flowers")
# importing dandelions

# ../input/sample/images/
SOURCE_IMAGES = os.path.join(PATH, "dandelion")

# ../input/sample/images/*.jpg
dandelion = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

# Prepare labels for dandelion

i = 0
while i < len(dandelion):
    labels.append("dandelion")
    flowers.append(dandelion[i])
    i += 1
    total += 1
    
print("Imported "+ str(i) + " images of dandelion flowers")
# importing roses

# ../input/sample/images/
SOURCE_IMAGES = os.path.join(PATH, "rose")

# ../input/sample/images/*.jpg
rose = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

# Prepare labels for daisy

i = 0
while i < len(rose):
    labels.append("rose")
    flowers.append(rose[i])
    i += 1
    total += 1

print("Imported "+ str(i) + " images of rose flowers")
# importing sunflower

# ../input/sample/images/
SOURCE_IMAGES = os.path.join(PATH, "sunflower")

# ../input/sample/images/*.jpg
sunflower = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

# Prepare labels for daisy

i = 0
while i < len(sunflower):
    labels.append("sunflower")
    flowers.append(sunflower[i])
    i += 1
    total += 1

print("Imported "+ str(i) + " images of sunflowers flowers")
# importing tulips

# ../input/sample/images/
SOURCE_IMAGES = os.path.join(PATH, "tulip")

# ../input/sample/images/*.jpg
tulip = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

# Prepare labels for daisy

i = 0
while i < len(tulip):
    labels.append("tulip")
    flowers.append(tulip[i])
    i += 1
    total += 1

print("Imported "+ str(i) + " images of tulip flowers")
print("Totals:")
print("Imported " + str(total) + " images in total")
def proc_images():

    x = [] # images as arrays
    y = [] # labels Infiltration or Not_infiltration
    WIDTH = 240
    HEIGHT = 240
    n = 0

    for img in flowers:
        base = os.path.basename(img)

        # Read and resize image
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        y.append(labels[n])
        n += 1

    print("Resized "+ str(n) + " images")
    return x,y 


print("Resizing Images...")
x,y = proc_images()
# Set it up as a dataframe if you like
df = pd.DataFrame()
df["labels"]=y
df["flowers"]=x
# Matplotlib black magic
print("Saving flower datasets as a NPZ files...")
np.savez("flowers", x)
np.savez("labels", y)         
    


!ls -1