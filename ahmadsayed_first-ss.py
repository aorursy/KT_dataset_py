import numpy as np 

import pandas as pd 

from tqdm import tqdm

import cv2

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import selectivesearch as ss

import os

%matplotlib inline

inputDataTest = []

output_imges = []
TEST_DIR = "../input/ade20k_2017_05_30_consistency/ADE20K_2017_05_30_consistency/images/consistencyanalysis"

def get_image_from_folder():

    for folder in tqdm(os.listdir(TEST_DIR)):

        new_path = os.path.join(TEST_DIR, folder)

        for img_name in tqdm(os.listdir(new_path)):

            if "_seg" not in img_name:

                if ".txt" not in img_name:

                    if "parts" not in img_name:

                        img_path = os.path.join(new_path, img_name)

                        img = cv2.imread(img_path)[:, :, ::-1]

                        inputDataTest.append(img)
get_image_from_folder()

for img in tqdm(inputDataTest):

    plt.imshow(img)

    plt.show()

    l, regions = ss.selective_search(img, scale=500, sigma=0.9, min_size=10)

    print("labels shape :", l.shape)

    new_img = img

    for i in range(l.shape[0]):

        for j in range(l.shape[1]):

            new_img[i][j] = l[i][j][-1]

        

    output_imges.append(new_img)

    plt.imshow(new_img)

    plt.show()

    
np.save("outputesImages.npy", output_imges)
    