import pandas as pd

import numpy as np

import cv2

import matplotlib.pyplot as plt
#Define RGB ranges

col_min = {"red": np.array([160, 0, 0]),

           "magenta": np.array([200, 0, 200]),

           "brown": np.array([75, 40, 10]),

           "blue": np.array([0, 40, 150]),

           "green": np.array([40, 140, 25])

           }



col_max = {"red": np.array([255, 50, 50]),

           "magenta": np.array([255, 55, 255]),

           "brown": np.array([130, 55, 20]),

           "blue": np.array([40, 80, 255]),

           "green": np.array([65, 255, 50])

           }



col_avg = {"red": 28,

           "magenta": 20,

           "brown": 12,

           "blue": 13,

           "green": 15

           }



#Function to count pixels in RGB range

def _count_dots(img, colors = ['red', 'magenta', 'brown', 'blue', 'green']):

    cnt = []

    for col in colors:

        cmsk = cv2.inRange(img, col_min[col], col_max[col])

        num = int(np.sum(cmsk > 0) / col_avg[col])

        cnt.append(num)

        print(col + ': ' + str(num))

#Test with image 0

id = 0

img = cv2.cvtColor(cv2.imread("../input/TrainDotted/" + str(id) + ".jpg"), cv2.COLOR_BGR2RGB)



#First crop

img_crop = img[1200:1800, 2200:2800]

plt.imshow(img_crop)

_count_dots(img_crop)



print("actual counts should be 4, 1, 10, 1, 10")
#Another crop

img_crop = img[1800:2400, 2800:3400]

plt.imshow(img_crop)

_count_dots(img_crop)
#Full image and compare to actual in train.csv

train = pd.read_csv("../input/Train/train.csv")

print(train.loc[id,:])

_count_dots(img)