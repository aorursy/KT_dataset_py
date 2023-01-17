# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import os

import cv2

import skimage

import matplotlib.pyplot as plt

from glob import glob

from PIL import Image, ImageOps

from scipy.misc import imresize

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer



Base_Data_Folder = "../input"

Train_Data_Folder = os.path.join(Base_Data_Folder, "train")
images = glob('../input/train/*/*.png')

images_per_class = {}

for class_folder_name in os.listdir(Train_Data_Folder):

    class_folder_path = os.path.join(Train_Data_Folder, class_folder_name)

    class_label = class_folder_name

    images_per_class[class_label] = []

    for image_path in glob(os.path.join(class_folder_path, "*.png")):

        image_rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        images_per_class[class_label].append(image_bgr)
for key,value in images_per_class.items():

    print("{0} -> {1}".format(key, len(value)))
# Test image to see the changes

test_1 = images_per_class["Black-grass"][97]

test_2 = images_per_class["Common wheat"][97]

test_3 = images_per_class["Loose Silky-bent"][97]



test_1hsv = cv2.cvtColor(test_1, cv2.COLOR_BGR2HSV)

test_2hsv = cv2.cvtColor(test_2, cv2.COLOR_BGR2HSV)

test_3hsv = cv2.cvtColor(test_3, cv2.COLOR_BGR2HSV)



sensitivity = 35

lower_hsv = np.array([60 - sensitivity, 100, 50])

upper_hsv = np.array([60 + sensitivity, 255, 255])



mask1 = cv2.inRange(test_1hsv, lower_hsv, upper_hsv)

mask2 = cv2.inRange(test_2hsv, lower_hsv, upper_hsv)

mask3 = cv2.inRange(test_3hsv, lower_hsv, upper_hsv)



kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))

mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel)



output1 = cv2.bitwise_and(test_1, test_1, mask = mask1)

output2 = cv2.bitwise_and(test_2, test_2, mask = mask2)

output3 = cv2.bitwise_and(test_3, test_3, mask = mask3)



output_blurred1 = cv2.GaussianBlur(output1, (0, 0), 3)

output_blurred2 = cv2.GaussianBlur(output2, (0, 0), 3)

output_blurred3 = cv2.GaussianBlur(output3, (0, 0), 3)



output_sharp1 = cv2.addWeighted(output1, 1.5, output_blurred1, -0.5, 0)

output_sharp2 = cv2.addWeighted(output2, 1.5, output_blurred2, -0.5, 0)

output_sharp3 = cv2.addWeighted(output3, 1.5, output_blurred3, -0.5, 0)



fig, axs = plt.subplots(1, 4, figsize=(20, 20))

axs[0].imshow(test_1)

axs[1].imshow(mask1)

axs[2].imshow(output1)

axs[3].imshow(output_sharp1)



fig, axs = plt.subplots(1, 4, figsize=(20, 20))

axs[0].imshow(test_2)

axs[1].imshow(mask2)

axs[2].imshow(output2)

axs[3].imshow(output_sharp2)



fig, axs = plt.subplots(1, 4, figsize=(20, 20))

axs[0].imshow(test_3)

axs[1].imshow(mask3)

axs[2].imshow(output3)

axs[3].imshow(output_sharp3)
sensitivity = 35

lower_hsv = np.array([60 - sensitivity, 100, 50])

upper_hsv = np.array([60 + sensitivity, 255, 255])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))



images_per_class_processed = {}

for class_folder_name in os.listdir(Train_Data_Folder):

    class_folder_path = os.path.join(Train_Data_Folder, class_folder_name)

    class_label = class_folder_name

    images_per_class_processed[class_label] = []

    for image_path in glob(os.path.join(class_folder_path, "*.png")):

        image_rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        image_resized = cv2.resize(image_bgr, (128, 128)) 

        image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

        

        mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)



        output = cv2.bitwise_and(image_resized, image_resized, mask = mask)



        output_blurred = cv2.GaussianBlur(output, (0, 0), 3)

        output_sharp = cv2.addWeighted(output, 1.5, output_blurred, -0.5, 0)

        

        images_per_class_processed[class_label].append(output_sharp)
sensitivity = 35

lower_hsv = np.array([60 - sensitivity, 100, 50])

upper_hsv = np.array([60 + sensitivity, 255, 255])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))



imagez = glob('../input/train/*/*.png')

labels = []

images_processed = []

for image in imagez:

    if image[-3:] != 'png':

        continue

    labels.append(image.split('/')[-2])

    new_img = Image.open(image)

    images_processed.append(ImageOps.fit(new_img, (128, 128), Image.ANTIALIAS).convert('RGB'))
sensitivity = 35

lower_hsv = np.array([60 - sensitivity, 100, 50])

upper_hsv = np.array([60 + sensitivity, 255, 255])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))



image = np.array(images_processed[213])

image_x = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

image_hsv = cv2.cvtColor(image_x, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

output = cv2.bitwise_and(image, image, mask = mask)

output_blurred = cv2.GaussianBlur(output, (0, 0), 3)

output_sharp = cv2.addWeighted(output, 1.5, output_blurred, -0.5, 0)

Image.fromarray(output_sharp)