# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
import random
import matplotlib.pyplot as plt
import pytesseract
%matplotlib inline
# print(cv2.__version__)
image = cv2.imread('/kaggle/input/rahulk/car_6.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def plot_images(img1, img2):
    fig = plt.figure(figsize=[20,20])
    ax1 = fig.add_subplot(121)     # one row two columns and targeting the first column
    ax1.imshow(img1)
    ax1.set(title="Normal Image")   # adding title

    ax2 = fig.add_subplot(122)    # one row two columns and targeting the second column
    ax2.imshow(img2, cmap='gray')
    ax2.set(title="Grayscale Image")  # adding title
plot_images(image, gray)
blur = cv2.bilateralFilter(gray, 10, 100, 100) # The 2nd parameter is used for blurring intensity..
def plot_images(img1, img2):
    fig = plt.figure(figsize=[20,20])
    ax1 = fig.add_subplot(121)     # one row two columns and targeting the first column
    ax1.imshow(img1,cmap='gray')
    ax1.set(title="Grayscale Image")   # adding title

    ax2 = fig.add_subplot(122)    # one row two columns and targeting the second column
    ax2.imshow(img2, cmap='gray')
    ax2.set(title="Grayscale Blurred Image")  # adding title
plot_images(gray, blur)
edges = cv2.Canny(blur, 15, 100)
def plot_images(img1, img2):
    fig = plt.figure(figsize=[20,20])
    ax1 = fig.add_subplot(121)     # one row two columns and targeting the first column
    ax1.imshow(img1,cmap='gray')
    ax1.set(title="Grayscale Blurred Image")   # adding title

    ax2 = fig.add_subplot(122)     # one row two columns and targeting the second column
    ax2.imshow(img2, cmap='gray')
    ax2.set(title="Image with Edges")  # adding title
plot_images(blur, edges)
counters, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #See, there are three arguments in cv.findContours() function, first one is source image, second is contour retrieval mode, third is contour approximation method.
# print(cnts)
# print(new)
image_copy = image.copy()
_ = cv2.drawContours(image_copy, counters, -1, (255,0,255),2)
def plot_images(img1, img2):
    fig = plt.figure(figsize=[20,20])
    ax1 = fig.add_subplot(121)     # one row two columns and targeting the first column
    ax1.imshow(img1,cmap='gray')
    ax1.set(title="Grayscale Blurred Image")   # adding title

    ax2 = fig.add_subplot(122)     # one row two columns and targeting the second column
    ax2.imshow(img2, cmap='gray')
    ax2.set(title="Image with Contours")      # adding title
plot_images(edges, image_copy)
print(len(counters))
counters_new = sorted(counters, key=cv2.contourArea, reverse=True)[:20] 
image_copy = image.copy()
_ = cv2.drawContours(image_copy, counters_new, -1, (255,0,255),2)
def plot_images(img1, img2):
    fig = plt.figure(figsize=[20,20])
    ax1 = fig.add_subplot(121)     # one row two columns and targeting the first column
    ax1.imshow(img1,cmap='gray')
    ax1.set(title="Grayscale Image")   # adding title

    ax2 = fig.add_subplot(122)     # one row two columns and targeting the second column
    ax2.imshow(img2, cmap='gray')
    ax2.set(title="Image with top 20 Contours")      # adding title
plot_images(image, image_copy)
plate = None

for counter in counters_new:
    perimeter = cv2.arcLength(counter, True)
    edges_count = cv2.approxPolyDP(counter, 0.02 * perimeter, True) # helps in counting the number of edges in an image
    if len(edges_count) == 4:
        x,y,w,h = cv2.boundingRect(counter)
        plate = image[y:y+h, x:x+w]
        break

cv2.imwrite("plate.png", plate)
def plot_images(img1):
    fig = plt.figure(figsize=[10,10])
    ax1 = fig.add_subplot(111)     # one row two columns and targeting the first column
    ax1.imshow(img1,cmap='gray')
    ax1.set(title="Plate Image")   # adding title  
plot_images(plate)
car_number = pytesseract.image_to_string(plate, lang="eng")
print(car_number)