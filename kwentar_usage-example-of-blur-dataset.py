import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import random

import cv2

import os

from tqdm import tqdm

import numpy as np

from sklearn.metrics import roc_auc_score
fig, axarr = plt.subplots(3,3)

fig.set_size_inches((20, 20))



src_dir = "../input/blur_dataset_scaled"

images = os.listdir(os.path.join(src_dir, 'sharp'))



for row_index in range(3):

    image = random.choice(images)

    print(image)

    for col_index, image_type in enumerate(['S', 'F', 'M']):

        if image_type == 'S':

            sub_dir = 'sharp'

        elif image_type == 'F':

            sub_dir = 'defocused_blurred'

        elif image_type == 'M':

            sub_dir = 'motion_blurred'

        image = image.split('.')[0][:-1] + image_type + '.' + image.split('.')[1]

        image_path = os.path.join(src_dir, sub_dir, image)

        image_data = mpimg.imread(image_path)

        axarr[row_index,col_index].imshow(image_data, interpolation='nearest')

        axarr[row_index,col_index].axis('off')

plt.show()
def variance_of_laplacian(image):

    return cv2.Laplacian(image, cv2.CV_64F).var()
src_dir = "../input/blur_dataset_scaled/sharp"

sharp_images = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]

sharp_values = []

for image_path in tqdm(sharp_images):

    image = cv2.imread(image_path)

    sharp_values.append(variance_of_laplacian(image))
src_dir = "../input/blur_dataset_scaled/defocused_blurred"

defocused_images = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]

defocused_values = []

for image_path in tqdm(defocused_images):

    image = cv2.imread(image_path)

    defocused_values.append(variance_of_laplacian(image))


src_dir = "../input/blur_dataset_scaled/motion_blurred"

motion_images = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]

motion_values = []

for image_path in tqdm(motion_images):

    image = cv2.imread(image_path)

    motion_values.append(variance_of_laplacian(image))
plt.hist(sharp_values, alpha=0.33, bins=20, color='r', range=(0,500), label='sharp')  # set x limit because in sharp we have values more than 5k

plt.hist(defocused_values, alpha=0.33, bins=20, color='b', range=(0,500), label='defocused')

plt.hist(motion_values, alpha=0.33, bins=20, color='g', range=(0,500), label='motion')

plt.legend()

plt.show()
labels = np.zeros(len(sharp_images)*3)

all_values = np.zeros(len(sharp_images)*3)

for index, v in enumerate(sharp_values):

    labels[index] = 1

    all_values[index] = v

for index, v in enumerate(defocused_values):

    all_values[index + len(sharp_images)] = v

for index, v in enumerate(motion_values):

    all_values[index + len(sharp_images)*2] = v

    

print(len(labels), sum(labels))

print(len(all_values), max(all_values), min(all_values), np.average(all_values))



print(f'ROC-AUC Value is {roc_auc_score(labels, all_values):.3f}')