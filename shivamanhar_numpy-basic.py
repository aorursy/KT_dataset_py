import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
arr = np.array([[[3, 4, 5], [8, 2, 5], [1, 9, 5],[3, 4, 5], [8, 2, 5], [1, 9, 5]],

               [[3, 0, 1], [0, 0, 0], [1, 10, 3],[3, 4, 5], [8, 20, 5], [1, 9, 5]],

               [[35, 50, 51], [8, 52, 57], [20, 20, 20],[3, 4, 11], [12, 2, 5], [1, 9, 5]],

               [[3, 0, 1], [8, 2, 7], [1, 10, 3],[3, 4, 5], [14, 2, 20], [1, 9, 5]],

               [[3, 0, 1], [8, 2, 7], [20, 20, 21],[3, 4, 5], [16, 2, 18], [1, 9, 5]],

               [[3, 0, 1], [13, 19, 16], [1, 10, 3],[5, 4, 6], [9, 2, 10], [1, 9, 5]]])

arr.shape
plt.imshow(arr)

plt.show()
arr[:,:,0] *= 10 
plt.imshow(arr)

plt.show()
arr[:,:,1] *= 7
plt.imshow(arr)

plt.show()
arr1 = np.copy(arr)
np.random.seed(5)

np.random.shuffle(arr1)

plt.imshow(arr1)

plt.show()
_,(ax) = plt.subplots(ncols=3, figsize=(16,5)) 

ax[0].hist(arr1[:,:,0],bins=10)

ax[1].hist(arr1[:,:,1],bins=10)

ax[2].hist(arr1[:,:,2],bins=10)

plt.show()
arr2 = arr1[:,:,0]+arr1[:,:,0]

plt.imshow(arr2)

plt.show()
red1 = np.concatenate((arr1[:,:,0], arr1[:,:,0]), axis=0)

green1 = np.concatenate((arr1[:,:,1], arr1[:,:,1]), axis=0)

blue1 = np.concatenate((arr1[:,:,2], arr1[:,:,2]), axis=0)
red1
_,(ax) = plt.subplots(ncols=3, figsize=(16,5)) 

ax[0].imshow(red1)

ax[1].imshow(green1)

ax[2].imshow(blue1)

plt.show()
image = cv2.merge((red1,green1, blue1))

plt.imshow(image)

plt.show()
image_rotation = np.rot90(image)

plt.imshow(image_rotation)

plt.show()
image_flip = np.fliplr(image)

plt.imshow(image_flip)

plt.show()
img_rec = np.copy(image)
img_rec[1,:,0] = 0

img_rec[1,:,1] = 50

img_rec[1,:,2] = 150
plt.imshow(img_rec)

plt.show()
img_rec[:,3,0] = 0

img_rec[:,3,1] = 50

img_rec[:,3,2] = 150

plt.imshow(img_rec)

plt.show()
ogrid_x, ogrid_y = np.ogrid[0:10, 0:5]
ogrid_x
ogrid_img = ogrid_x+ogrid_y

print(ogrid_img)

plt.imshow(ogrid_img)

plt.show()
ogrid_img = ogrid_img.T

plt.imshow(ogrid_img)

plt.show()
YY, XX = np.mgrid[10:40:10, 1:4]

ZZ = XX + YY 

ZZ
circle_mask = 5**2 + 7**2 <= 100**2

circle_mask
demo_image = plt.imread('/kaggle/input/sample-images-for-kaggle-demos/1928768_1035869614877_9398_n.jpg')

plt.imshow(demo_image)

plt.show()
# Get the dimensions

n,m,d = demo_image.shape

print(n,m,d)
# Create an open grid for our image

x,y = np.ogrid[0:n,0:m]
#copy image

copyImg = demo_image.copy()



#get the x and y center points of our image

center_x = n/2

center_y = m/2

print("Center x and Center y", center_x, center_y)



#create a circle mask which is centered in the middle of the image

circle_mask = (x-center_x)**2+(y-center_y)**2 <= 8000



copyImg[circle_mask] = [0, 0,0]



plt.imshow(copyImg)

plt.show()

square_mask = (x<200)&(x>100)&(y<500)&(y>400)



copyImg[square_mask] = [255, 0,0]



plt.imshow(copyImg)

plt.show()
copyImg = demo_image.copy()



copyImg = demo_image[x, -y]



plt.imshow(copyImg)

plt.show()
copyImg = demo_image.copy()



copyImg = demo_image[-x, y]



plt.imshow(copyImg)

plt.show()