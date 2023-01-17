import numpy as np

import cv2

import matplotlib.pyplot as plt

import os

from tqdm import tqdm

from sklearn.cluster import MeanShift, estimate_bandwidth





TEST_DIR = "../input/ade20k_2017_05_30_consistency/ADE20K_2017_05_30_consistency/images/consistencyanalysis"



inputDataTest = []







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

def apply_mean_shift(data):

    segment_imgs = []

    for img in tqdm(data):

        initial_shape = img.shape

        print("image shape : ", initial_shape)

        flat_image = np.reshape(img, [-1, 3])

        print("flat image shape :", flat_image.shape)



        bandwidth2 = estimate_bandwidth(flat_image,

                                        quantile=.2, n_samples=500)

        ms = MeanShift(bandwidth2, bin_seeding=True)

        ms.fit(flat_image)

        labels = ms.labels_



        plt.figure(2)

        plt.subplot(2, 1, 1)

        plt.imshow(img)

        plt.axis('off')

        plt.subplot(2, 1, 2)

        segmented_image = np.reshape(labels, initial_shape[0:2])

        segment_imgs.append(segmented_image)

        plt.imshow(segmented_image)

        plt.axis('off')

        plt.show()

    return segment_imgs

segImgs = apply_mean_shift(inputDataTest)

print("Saveing images")

np.save("images.npy", segImgs)

print("Saveing is done")