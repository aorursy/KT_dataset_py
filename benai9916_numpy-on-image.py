import numpy as np
my_list = [1,2,3,4]

type(my_list)
# make a numpy array
np_array = np.array(my_list)
np_array
type(np_array)
# createa an array

np.arange(1,10)
# create all zeros
np.zeros(4)
# crate 2d array with zeros
np.zeros(shape=(3,5))
# create 1's array
np.ones(10)
# 1's 2D matrix

np.ones((4,5))
# generate random number
np.random.seed(23)
arr = np.random.randint(0, 100, 20)
arr
# get max value
np.max(arr)
# return the index loaction of 91
np.argmax(arr)
# get min value

np.min(arr)
# get the index of min value
np.argmin(arr)
# reshape an array
arr = arr.reshape((4,5))
arr
# get the element by index (slicing)
# arr(row, column)

arr[0,3]
# row zero to 1 and third column
arr[0:2, 3]
# slice and reshape
arr[:, 1].reshape(2,2)
# reaplace the element 

arr[1, :] = 0
arr
# copy an array

new_arr = arr.copy()

new_arr
# mutiply
new_arr[0, :] * 2
from PIL import Image

# show image
import matplotlib.pyplot as plt
img = Image.open('../input/opencv-samples-images/data/lena.jpg')

type(img)
# cnvert to numpy array
img = np.asarray(img)

type(img)
# converted to 3 dimensional array
img.ndim
# no of row, column and, dimension
img.shape
img
# show the image
plt.imshow(img)
plt.show()
# copy the image

new_img = img.copy()

plt.imshow(new_img)
#  R G B
# display only red channel

red_img = new_img[:,:, 0]

plt.imshow(red_img)
# display only green channel

green_img = new_img[:,:, 2]

plt.imshow(green_img)
# display only blue channel

blue_img = new_img[:,:, 0]

plt.imshow(blue_img)
# image in gray

plt.imshow(new_img[:,:, 0], cmap='gray')
# remove red channel

new_img[:,:, 0] = 0

plt.imshow(new_img)
# remove green channel

new_img[:,:, 1] = 0

plt.imshow(new_img)
# red and green channel to 0
new_img[:,:, 0:2] = 0
plt.imshow(new_img)
img2 = img.copy()
img2[:,:, 0:2] = 0

plt.imshow(img2)
