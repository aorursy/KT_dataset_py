import cv2

import numpy as np

import matplotlib.pyplot as plt

from skimage import io,exposure,morphology

from skimage.filters import gaussian

from skimage import feature

import pandas as pd

import os

from os import path

from scipy.signal import convolve



%matplotlib inline
path.isfile(r"../input/bettercsv/Horizontal_Single_frame.csv")
files = os.listdir(r"../input/bettercsv/")
(files)
file = ['Horizontal_Single_frame_dB.csv',

 'Diagonal_Single_frame_dB.csv',

 'Random_Single_frame_dB.csv',

 'Vertical_Single_frame_dB.csv']
img = pd.read_csv(r"../input/bettercsv/Horizontal_Single_frame.csv",header = None)
img
img = np.array(img)
h,w = img.shape
img = exposure.equalize_hist(img)

img = (img - np.min(img))/(np.max(img) - np.min(img))
plt.imshow(img,cmap = 'gray')

plt.title("Original img")

plt.axis("off")

plt.show()
plt.imsave('acc.png',img)
sigma = 0.10





fil = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

y = convolve(img, np.flip(fil.T, axis=0))

x = convolve(img, fil)



gradient_magnitude = np.sqrt(np.square(x) + np.square(y))

gradient_magnitude *= 255.0 / gradient_magnitude.max()



plt.imshow(gradient_magnitude)

plt.axis("on")



plt.title("Edge detected images")

plt.show()
gradient_magnitude
gray_filtered = cv2.bilateralFilter(np.uint8(img), 7, 50, 50)



kernel = np.ones((3,3),np.uint8)

d = cv2.erode(gray_filtered,kernel,iterations = 2)



plt.imshow(d)

#plt.xlim(-5,5)

plt.axis("on")





plt.title("Edge detected images")

plt.show()
j = 0
for i in (file):

    

    print(str(i))

    img = pd.read_csv(r"../input/bettercsv/" +str(i),header = None)

    img = np.array(img)

    img = exposure.equalize_hist(img)

    img = (img - np.min(img))/(np.max(img) - np.min(img))



    sigma = 0.10



    fil = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    y = convolve(img, np.flip(fil.T, axis=0))

    x = convolve(img, fil)



    gradient_magnitude = np.sqrt(np.square(x) + np.square(y))

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    

    d = feature.canny(img, sigma=3)

    

    plt.imsave('sobel' + str(j) + '.png',gradient_magnitude)

    plt.imsave('canny' + str(j) + '.png',d)

    plt.imsave('ori' + str(j) + '.png',img)

    

    i1 = plt.imshow(img)

    plt.title("Original image")

    plt.xlabel("Azimuth ")

    plt.ylabel("Elevation ")

    plt.axis("off")

    plt.colorbar(i1)

    plt.show()



    i2 = plt.imshow(gradient_magnitude)

    plt.axis("off")

    plt.xlabel("Azimuth ")

    plt.ylabel("Elevation ")

    plt.title("After applying Sobel filter")

    plt.colorbar(i2)

    plt.show()



    i3 = plt.imshow(d)

    plt.axis("off")

    plt.xlabel("Azimuth ")

    plt.ylabel("Elevation ")

    plt.title("After applying Canny filter")

    plt.colorbar(i3)

    plt.show()

    

    j = j + 1
img = pd.read_csv(r"../input/bettercsv/Vertical_Single_frame.csv",header = None)

img = np.array(img)

img = exposure.equalize_hist(img)

img = (img - np.min(img))/(np.max(img) - np.min(img))



sigma = 0.10



fil = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

y = convolve(img, np.flip(fil.T, axis=0))

x = convolve(img, fil)



gradient_magnitude = np.sqrt(np.square(x) + np.square(y))

gradient_magnitude *= 255.0 / gradient_magnitude.max()



plt.subplot(121),plt.imshow(img,cmap = 'binary')

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(gradient_magnitude,cmap = 'gray')

plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
np.unique(gradient_magnitude)
plt.imsave('sobel_hori.png',gradient_magnitude)
img = pd.read_csv(r"../input/bettercsv/Vertical_Single_frame.csv",header = None)

img = np.array(img)

img = exposure.equalize_hist(img)



v = np.median(img)

l = int(max(0, (1.0 - sigma) * v))

u = int(min(255, (1.0 + sigma) * v))

e = feature.canny(img, sigma=3)





plt.subplot(121),plt.imshow(img,cmap = 'binary')

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(e,cmap = 'gray')

plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
np.unique(e)
plt.imsave('canny_hori.png',e)
'''kernel = np.ones((5,5)) 

er = morphology.erosion(edges, kernel)'''
'''plt.imshow(edges,cmap = 'binary_r')

plt.title("After Canny Edge")

plt.axis("off")

plt.show()'''
'''plt.imshow(er,cmap = 'binary_r')

plt.title("Binary marked image")

plt.axis("off")

plt.show()'''
from skimage import filters

image = filters.sobel(img)
#hist = filters.apply_hysteresis_threshold(gradient_magnitude, l, u)
#plt.imsave('hister_hori_skimg.png',hist)
#plt.imsave('sobel_hori_skimg.png',image)
import numpy as np

import scipy.stats as st



def gkern(l, sig):

    """\

    creates gaussian kernel with side length l and a sigma of sig

    """



    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)

    xx, yy = np.meshgrid(ax, ax)



    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))



    return kernel / np.sum(kernel)
import pandas as pd
df = pd.DataFrame(gkern(9,3))
df.to_csv('9*9_gauss.csv')
#backtorgb = cv2.cvtColor(d,cv2.COLOR_GRAY2RGB)
import random as rng

img = pd.read_csv(r"../input/bettercsv/Random_Single_frame.csv",header = None)

img = np.array(img)

img = exposure.equalize_hist(img)



v = np.median(img)

l = int(max(0, (1.0 - sigma) * v))

u = int(min(255, (1.0 + sigma) * v))

d = feature.canny(img, sigma=3)
# Find contours

contours,_ = cv2.findContours(np.uint8(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the convex hull object for each contour

hull_list = []

for i in range(len(contours)):

    hull = cv2.convexHull(contours[i])

    hull_list.append(hull)

# Draw contours + hull results

drawing = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)

for i in range(len(contours)):

    color = (255,0,0)

    cv2.drawContours(drawing, contours, i, color)

    cv2.drawContours(drawing, hull_list, i, color)
plt.imsave('main_contour.png',drawing)
plt.imshow(drawing)

plt.title("Binary marked image")

plt.axis("off")

plt.show()
contours
# Find contours

contours,_ = cv2.findContours(np.uint8(d), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the convex hull object for each contour

hull_list = []

for i in range(len(contours)):

    hull = cv2.convexHull(contours[i])

    hull_list.append(hull)

# Draw contours + hull results

drawing = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)

for i in range(len(contours)):

    color = (255,0,0)

    cv2.drawContours(drawing, contours, i, color)

    cv2.drawContours(drawing, hull_list, i, color)
plt.imsave('detected_contour.png',drawing)
'''x_arr = []

y_arr = []



for i in range(4):

    for j in range(77):

        x_arr.append(contours[i][j][0][0])

        y_arr.append(contours[i][j][0][1])'''
#c.shape
plt.imshow(d)

plt.title("Binary marked image")

plt.axis("off")

plt.show()
plt.imshow(drawing)

plt.title("Binary marked detected image")

plt.axis("off")

plt.show()