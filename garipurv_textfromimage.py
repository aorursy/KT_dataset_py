import os

import cv2

import numpy as np

import matplotlib.pyplot as plt



for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        image = cv2.imread(os.path.join(dirname, filename))



hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)



lower_value = np.array([0,0,0])

upper_value = np.array([200,255,80])

mask = cv2.bitwise_not(cv2.inRange(hsv, lower_value, upper_value))



kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (10,10))

opening_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)



opening_mask = cv2.cvtColor(opening_mask, cv2.COLOR_RGB2BGR)

plt.imshow(opening_mask)