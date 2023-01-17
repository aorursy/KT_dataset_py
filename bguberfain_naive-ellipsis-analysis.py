import numpy as np

import pandas as pd

import random

import skimage.io

import matplotlib.pyplot as plt

import os

from skimage import morphology

import cv2

from tqdm import tqdm

import seaborn as sns



%matplotlib inline
img_ids = os.listdir("../input/stage1_train/")
# This will [safelly] find the parameters of fitted ellipse

def find_ellipse(mask):

    try:

        ret, thresh = cv2.threshold(mask, 127, 255, 0)

        _, contours, hierarchy = cv2.findContours(thresh, 1, 2)

        cnt = contours[0]

        

        ellipse = cv2.fitEllipse(cnt)

    except:

        ellipse = (1, 1), (1, 1), 1

    

    return ellipse
# This will analyse the fitted ellipsis of an image

def ellipse_analysis(image_id):

    # Read files

    image_file = "../input/stage1_train/{}/images/{}.png".format(image_id,image_id)

    mask_file = "../input/stage1_train/{}/masks/*.png".format(image_id)

    

    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)

    masks = skimage.io.imread_collection(mask_file).concatenate()

    

    # Find mask ellipsis

    ellipsis = list(map(find_ellipse, masks))

    

    # Find width and height (second parameter)

    ellipse_wh = lambda e: e[1]

    ellipsis_wh = np.array(list(map(ellipse_wh, ellipsis)))

    

    # Calculate rate of each ellipisis

    rates = ellipsis_wh.min(axis=1)/ellipsis_wh.max(axis=1)

    

    # Calculate mean RGB of image

    img_mean_rgb = image.mean(axis=(0, 1))

    

    return rates.mean(), rates.std(), len(ellipsis), img_mean_rgb[0], img_mean_rgb[1], img_mean_rgb[2]
# Dataframe for results (columns must sync with ellipse_analysis return)

df_results = pd.DataFrame(index=img_ids, dtype=np.float,

                          columns=["ellipsis_ratio_mean", "ellipsis_ratio_std",

                                   "n_ellipsis", "image_mean_red", "image_mean_green",

                                   "image_mean_blue"])

# Start analysis

for image_id in tqdm(img_ids):

    df_results.loc[image_id] = ellipse_analysis(image_id)
# Plot results, with color being the average color of images

fig, ax = plt.subplots(figsize=(10, 10))

df_results.plot(kind="scatter", x="ellipsis_ratio_mean", y="ellipsis_ratio_std", ax=ax, s=50, edgecolor='gray', lw = 1,

        c=df_results[["image_mean_red", "image_mean_green", "image_mean_blue"]].values / 255);