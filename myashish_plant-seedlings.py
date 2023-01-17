%matplotlib inline

import os

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd

import cv2

import numpy as np

from glob import glob

import seaborn as sns
BASE_DATA_FOLDER = "../input"

TRAin_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, "train")
images_per_class = {}

for class_folder_name in os.listdir(TRAin_DATA_FOLDER):

    class_folder_path = os.path.join(TRAin_DATA_FOLDER, class_folder_name)

    class_label = class_folder_name

    images_per_class[class_label] = []

    for image_path in glob(os.path.join(class_folder_path, "*.png")):

        image_rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        images_per_class[class_label].append(image_bgr)
for key,value in images_per_class.items():

    print("{0} -> {1}".format(key, len(value)))
def plot_for_class(label):

    nb_rows = 3

    nb_cols = 3

    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(6, 6))



    n = 0

    for i in range(0, nb_rows):

        for j in range(0, nb_cols):

            axs[i, j].xaxis.set_ticklabels([])

            axs[i, j].yaxis.set_ticklabels([])

            axs[i, j].imshow(images_per_class[label][n])

            n += 1        
plot_for_class("Small-flowered Cranesbill")
plot_for_class("Maize")
def create_mask_for_plant(image):

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)



    sensitivity = 35

    lower_hsv = np.array([60 - sensitivity, 100, 50])

    upper_hsv = np.array([60 + sensitivity, 255, 255])



    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    

    return mask



def segment_plant(image):

    mask = create_mask_for_plant(image)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output



def sharpen_image(image):

    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)

    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)

    return image_sharp
# Test image to see the changes

image = images_per_class["Small-flowered Cranesbill"][97]



image_mask = create_mask_for_plant(image)

image_segmented = segment_plant(image)

image_sharpen = sharpen_image(image)



fig, axs = plt.subplots(1, 4, figsize=(20, 20))

axs[0].imshow(image)

axs[1].imshow(image_mask)

axs[2].imshow(image_segmented)

axs[3].imshow(image_sharpen)
def find_contours(mask_image):

    return cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]



def calculate_largest_contour_area(contours):

    if len(contours) == 0:

        return 0

    c = max(contours, key=cv2.contourArea)

    return cv2.contourArea(c)



def calculate_contours_area(contours, min_contour_area = 250):

    area = 0

    for c in contours:

        c_area = cv2.contourArea(c)

        if c_area >= min_contour_area:

            area += c_area

    return area
areas = []

larges_contour_areas = []

labels = []

nb_of_contours = []



for class_label in images_per_class.keys():

    for image in images_per_class[class_label]:

        mask = create_mask_for_plant(image)

        contours = find_contours(mask)

        

        area = calculate_contours_area(contours)

        largest_area = calculate_largest_contour_area(contours)

        

        areas.append(area)

        nb_of_contours.append(len(contours))

        larges_contour_areas.append(largest_area)

        labels.append(class_label)
features_df = pd.DataFrame()

features_df["label"] = labels

features_df["area"] = areas

features_df["largest_area"] = larges_contour_areas

features_df["number_of_components"] = nb_of_contours
features_df.groupby("label").describe()