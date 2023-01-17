import numpy as np 

import json 

import cv2

import os

from matplotlib import pyplot as plt

%matplotlib inline
DATA_DIR = '../input/car-plates-ocr/data/'

NUM_SAMPLES = 5



train_marks = json.load(open(os.path.join(DATA_DIR, 'train.json')))

sample_marks = np.random.choice(train_marks, size=NUM_SAMPLES)
def order_points(pts):

    

    rect = np.zeros((4, 2), dtype = "float32")

    

    s = pts.sum(axis = 1)

    rect[0] = pts[np.argmin(s)]

    rect[2] = pts[np.argmax(s)]

    

    diff = np.diff(pts, axis = 1)

    rect[1] = pts[np.argmin(diff)]

    rect[3] = pts[np.argmax(diff)]

    

    return rect





def four_point_transform(image, pts):

    

    rect = order_points(pts)

    

    tl, tr, br, bl = pts

    

    width_1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))

    width_2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    max_width = max(int(width_1), int(width_2))

    

    height_1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))

    height_2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    max_height = max(int(height_1), int(height_2))

    

    dst = np.array([

        [0, 0],

        [max_width, 0],

        [max_width, max_height],

        [0, max_height]], dtype = "float32")

    

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped
plt.figure(figsize=(25, 16))

for i, mark in enumerate(sample_marks):

    box = np.array(mark['nums'][0]['box'])

    image = cv2.imread(os.path.join(DATA_DIR, mark['file']))

    image = image[..., ::-1]

    

    

    plt.subplot(2, NUM_SAMPLES, i + 1)

    plt.imshow(image)

    plt.subplot(2, NUM_SAMPLES, i + NUM_SAMPLES + 1)

    plt.imshow(four_point_transform(image, box))



plt.tight_layout()

plt.show()