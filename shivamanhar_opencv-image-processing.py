# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

#from PIL import Image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
img1 = cv2.imread('/kaggle/input/image-processing/horse.jpg')



plt.imshow(img1)

plt.show()

img2 = cv2.imread('/kaggle/input/image-processing/cat.jpg')

plt.imshow(img2)

plt.show()

img3 = cv2.imread('/kaggle/input/image-processing/bird.jpg')

plt.imshow(img3)

plt.show()

img4 = cv2.imread('/kaggle/input/image-processing/flowers.jpg')

plt.imshow(img4)

plt.show()
img1
print( img1.shape )
len(img1.shape)
print(img1[150,200])
#img1

img2[:,:,1] = 0

img2[:,:,0] = 0

#plt.imshow(img2)



plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.show()
img_name =[]

img_height = []

img_width = []

img_channel =[]

r_value =[]

g_value =[]

b_value =[]

dir_path = '/kaggle/input/image-processing/'

for file_name in os.listdir(dir_path):

    img_name.append(file_name)

    img = cv2.imread(dir_path+'/'+file_name)

    img_height.append(img.shape[1])

    img_width.append(img.shape[0])

    img_channel.append(img.shape[2])

    x = int(img.shape[0]/2)

    y = int(img.shape[1]/2)

    r_value.append(img[x,y][0])

    g_value.append(img[x,y][1])

    b_value.append(img[x,y][2])

    

np.savetxt("stats.csv", np.column_stack((img_name, img_width, img_height, img_channel,r_value,g_value,b_value)), delimiter=",", fmt='%s')

#print(img_height)
plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))

plt.show()



b_channel, g_channel, r_channel = cv2.split(img4)

alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 

img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

cv2.imwrite('flowers_alpha.png',img_BGRA)

plt.imshow(cv2.cvtColor(img_BGRA, cv2.COLOR_BGR2BGRA))

plt.show()

img_BGRA.shape
img_BGRA[:,:,3] = 127
plt.imshow(cv2.cvtColor(img_BGRA, cv2.COLOR_BGR2RGB))

plt.show()
# First create the image with alpha channel

rgba = cv2.cvtColor(img4, cv2.COLOR_RGB2RGBA)



# Then assign the mask to the last channel of the image

rgba[:, :, 3] = 0.5

cv2.imwrite('result.png',rgba)
plt.imshow(cv2.cvtColor(rgba, cv2.COLOR_BGR2BGRA))

plt.show()
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

plt.show()



b_channel, g_channel, r_channel = cv2.split(img1)

horse_gray = ((0.3 * r_channel) + (0.59 * g_channel)+ (0.11 * b_channel))

#horse_gray = cv2.merge((b_channel, g_channel, r_channel))

print(horse_gray.shape)

plt.imshow(horse_gray)

plt.show()
rosebloom = cv2.VideoCapture('/kaggle/input/video-processing/RoseBloom.mp4')

type(rosebloom)

frame_width = int(rosebloom.get(3))

print(frame_width)

frame_height = int(rosebloom.get(4))

print(frame_height)
import cv2

import numpy as np

import os



dir_path = '/kaggle/input/image-processing/' #set image path

horse_img = cv2.imread(dir_path+'/horse.jpg')

#b_channel, g_channel, r_channel = cv2.split(horse_img)

r_channel= horse_img[:,:,0]

b_channel= horse_img[:,:,1]

g_channel= horse_img[:,:,2]

horse_gray = ((0.3 * r_channel) + (0.59 * g_channel)+ (0.11 * b_channel))

#horse_gray = cv2.merge((b_channel, g_channel, r_channel))

cv2.imwrite('horse_gray.jpg',horse_gray)