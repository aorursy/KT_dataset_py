# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/txisteak/12744721_1060598450673849_1876892865353905715_n-2.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
# Load our new image

image = cv2.imread('/kaggle/input/txisteak/10407529_703031933113077_8371381867608476511_n.jpg', 0)



plt.figure(figsize=(30, 30))

plt.subplot(3, 2, 1)

plt.title("Original")

plt.imshow(image)





#Values below 127 goes to 0 (black, everything above goes to 255 (white)

ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)



plt.subplot(3, 2, 2)

plt.title("Threshold Binary")

plt.imshow(thresh1)



# It's good practice to blur images as it removes noise

image = cv2.GaussianBlur(image, (3, 3), 0)



# Using adaptiveThreshold

thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5) 



plt.subplot(3, 2, 3)

plt.title("Adaptive Mean Thresholding")

plt.imshow(thresh)





_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



plt.subplot(3, 2, 4)

plt.title("Otsu's Thresholding")

plt.imshow(th2)





plt.subplot(3, 2, 5)

# Otsu's thresholding after Gaussian filtering

blur = cv2.GaussianBlur(image, (5,5), 0)

_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.title("Guassian Otsu's Thresholding")

plt.imshow(th3)

plt.show()