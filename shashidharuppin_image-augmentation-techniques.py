# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Reading the training dataset

training = pd.read_csv('/kaggle/input/training/training.csv')
training.info()
#Trying to find the number of null values in the training dataset
training.isnull().sum()
#Looks like we have lot of null values in the train data. Lets drop all the values which have null value.
training = training.dropna()
training.info()
#There are total 2140 non-null objects. Lets see the top 5 details of the image dataset.
training.head()
#We can see that the first 30 columns refers to the points/keypoints for images and last columns refers to the pixel values of the images.
# In order to proceed further we need to seperate the images and the datapoints from the training data.

#Writing a function to seperate the images from the training data
def load_images(image_data):
    images = []
    for idx, sample in image_data.iterrows():  #iterrows used to iterate over the dataframe
        image = np.array(sample['Image'].split(' '), dtype=int)
        image = np.reshape(image, (96,96,1))
        images.append(image)
    images = np.array(images)/255.             #Normalizing the image btw 0 and 1
    return images

#Writing a function to seperate the keypoints from the training data
def load_keypoints(keypoint_data):
    keypoint_data = keypoint_data.drop('Image',axis = 1)
    keypoint_features = []
    for idx, sample_keypoints in keypoint_data.iterrows():
        keypoint_features.append(sample_keypoints)
    keypoint_features = np.array(keypoint_features, dtype = 'float')
    return keypoint_features


clean_train_images =load_images(training)
print("Shape of clean_train_images:", np.shape(clean_train_images))

clean_train_keypoints = load_keypoints(training)
print("Shape of clean_train_keypoints:", np.shape(clean_train_keypoints))

#Below is the link for the more usage on iterrows
#https://cmdlinetips.com/2018/12/how-to-loop-through-pandas-rows-or-how-to-iterate-over-pandas-rows/
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img, img_to_array, load_img

flipped_images = np.flip(clean_train_images[0], axis=2) 

plt.imshow(array_to_img(flipped_images))
flipped_images = np.flip(clean_train_images[0], axis=1) 

plt.imshow(array_to_img(flipped_images))
flipped_images = np.flip(clean_train_images[0], axis=0) 

plt.imshow(array_to_img(flipped_images))
%matplotlib inline


for i in range(1,9):
    fig = plt.figure(figsize=(20,20))
    
    ax0 = fig.add_subplot(3,4,1)
    ax0.set_title("Original Image")
    ax0.imshow(array_to_img(clean_train_images[i]))
    
    flip1 = np.flip(clean_train_images[i], axis=1)
    ax1 = fig.add_subplot(3,4,2)
    ax1.set_title("Axis =1 Horizontal Left Flip")
    ax1.imshow(array_to_img(flip1))
    
    flip2 = np.flip(clean_train_images[i], axis=2)
    ax2 = fig.add_subplot(3,4,3)
    ax2.set_title("Axis =2 Horizontal Right Flip")
    ax2.imshow(array_to_img(flip2))
    
    flip3 = np.flip(clean_train_images[i], axis=0)
    ax3 = fig.add_subplot(3,4,4)
    ax3.set_title("Axis =0 Vertical Flip")
    ax3.imshow(array_to_img(flip3))
#Function for Horizantal flipping of images
def left_right_flip(images, keypoints):
    flipped_keypoints = []
    flipped_images = np.flip(images, axis=2)   # Flip column-wise (axis=2)
    for idx, sample_keypoints in enumerate(keypoints):
        flipped_keypoints.append([96.-coor if idx%2==0 else coor for idx,coor in enumerate(sample_keypoints)])    # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
    return flipped_images, flipped_keypoints
horizontal_flip = True
if horizontal_flip:
    flipped_train_images, flipped_train_keypoints = left_right_flip(clean_train_images, clean_train_keypoints)
    print("Shape of flipped_train_images:",np.shape(flipped_train_images))
    print("Shape of flipped_train_keypoints:",np.shape(flipped_train_keypoints))
#Let's create a function to plot the image
def plot_sample(image, keypoint, axis, title):
    image = image.reshape(96,96)
    axis.imshow(image, cmap='gray')
    axis.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20)
    plt.title(title)
    
print("************************************* Horizontal Flippied Images with Key Points*********************************************")
if horizontal_flip:
    fig = plt.figure(figsize=(20,8))
    for i in range(10):
        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        plot_sample(flipped_train_images[i], flipped_train_keypoints[i], axis, "Horizontal Flip Augmentation:")
    plt.show()

print("************************************ Original Images with Key Points ********************************************************")
if horizontal_flip:
    fig = plt.figure(figsize=(20,8))
    for i in range(10):
        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        plot_sample(clean_train_images[i], clean_train_keypoints[i], axis, "Original Image Points")
    plt.show()

# Above is the formula to get the new co-ordinates from the given co-ordinates with the rotation angle
import cv2
from math import sin, cos, pi


rotation_angles = [12]    # Rotation angle in degrees (includes both clockwise & anti-clockwise rotations)

#Writing a function for Rotation of the Images
def rotate_augmentation(images, keypoints):
    rotated_images = []
    rotated_keypoints = []
    print("Augmenting for angles (in degrees): ")
    
    for angle in rotation_angles:    # Rotation augmentation for a list of angle values
        for angle in [angle,-angle]:
            print(f'{angle}', end='  ')
            M = cv2.getRotationMatrix2D((48,48), angle, 1.0)
            angle_rad = -angle*pi/180.     # Obtain angle in radians from angle in degrees (notice negative sign for change in clockwise vs anti-clockwise directions from conventional rotation to cv2's image rotation)
            
            # For train_images
            for image in images:
                rotated_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
                rotated_images.append(rotated_image)
            
            # For train_keypoints
            for keypoint in keypoints:
                rotated_keypoint = keypoint - 48.    # Subtract the middle value of the image dimension
                for idx in range(0,len(rotated_keypoint),2):
                    # https://in.mathworks.com/matlabcentral/answers/93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point
                    rotated_keypoint[idx] = rotated_keypoint[idx]*cos(angle_rad)-rotated_keypoint[idx+1]*sin(angle_rad)
                    rotated_keypoint[idx+1] = rotated_keypoint[idx]*sin(angle_rad)+rotated_keypoint[idx+1]*cos(angle_rad)
                rotated_keypoint += 48.   # Add the earlier subtracted value
                rotated_keypoints.append(rotated_keypoint)
            
    return np.reshape(rotated_images,(-1,96,96,1)), rotated_keypoints

#For more details on the transformation of the images below is the link.
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html


rotation_augmentation = True

if rotation_augmentation:
    rotated_train_images, rotated_train_keypoints = rotate_augmentation(clean_train_images, clean_train_keypoints)
    print("\nShape of rotated_train_images:",np.shape(rotated_train_images))
    print("Shape of rotated_train_keypoints:\n",np.shape(rotated_train_keypoints))
#Rotation of Kep Points with Images using "imgaug" library. You can find an example on the below link.
#https://github.com/shashidharuppin/Deep-Learning/blob/master/Image%20Augmentation%20Rotation%20with%20Key%20points.ipynb
rotation_augmentation = True

print("************************************* Rotationally Flippied Images with Key Points*********************************************")
if rotation_augmentation:
    fig = plt.figure(figsize=(20,8))
    for i in range(10):
        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        plot_sample(rotated_train_images[i], rotated_train_keypoints[i], axis, "Rotation Augmentation:")
    plt.show()
#Writing a function for Brightness Alteration using Numpy Clip method
def alter_brightness(images, keypoints):
    altered_brightness_images = []
    inc_brightness_images = np.clip(images*1.2, 0.0, 1.0)    # Increased brightness by a factor of 1.2 & clip any values outside the range of [-1,1]
    dec_brightness_images = np.clip(images*0.6, 0.0, 1.0)    # Decreased brightness by a factor of 0.6 & clip any values outside the range of [-1,1]
    altered_brightness_images.extend(inc_brightness_images)
    altered_brightness_images.extend(dec_brightness_images)
    return altered_brightness_images, np.concatenate((keypoints, keypoints))


brightness_augmentation = True

if brightness_augmentation:
    altered_brightness_train_images, altered_brightness_train_keypoints = alter_brightness(clean_train_images, clean_train_keypoints)
    print("Shape of altered_brightness_train_images:",np.shape(altered_brightness_train_images))
    print("Shape of altered_brightness_train_keypoints:",np.shape(altered_brightness_train_keypoints))
print("************************************* Increased Brightness Images with Key Points*********************************************")
if brightness_augmentation:
    fig = plt.figure(figsize=(20,8))
    for i in range(10):
        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        plot_sample(altered_brightness_train_images[i], altered_brightness_train_keypoints[i], axis, "Brightness Augmentation: ")
    plt.show()
    
print("************************************* Decreased Brightness Images with Key Points*********************************************")
if brightness_augmentation:
    fig = plt.figure(figsize=(20,8))
    for i in range(2141,2151,1):
        axis = fig.add_subplot(2, 5, i-2140, xticks=[], yticks=[])
        plot_sample(altered_brightness_train_images[i], altered_brightness_train_keypoints[i], axis, "Brightness Augmentation: ")
    plt.show()
from PIL import Image,ImageEnhance

fig = plt.figure(figsize=(20,8))
print("************************************Increased Brightness Images**********************************************************")
for i in range(10):
    axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
    img = array_to_img(clean_train_images[i])
    img_brightness_obj=ImageEnhance.Brightness(img)
    factor = 1.2
    enhanced_img=img_brightness_obj.enhance(factor)
    plt.imshow(enhanced_img)
# Facial Keypoints remain the same


print("************************************Decreased Brightness Images**********************************************************")
fig1 = plt.figure(figsize=(20,8))
for i in range(10):
    axis1 = fig1.add_subplot(2, 5, i+1, xticks=[], yticks=[])
    img = array_to_img(clean_train_images[i])
    img_brightness_obj=ImageEnhance.Brightness(img)
    factor = 0.6
    enhanced_img=img_brightness_obj.enhance(factor)
    plt.imshow(enhanced_img)
#Writing a function to add noise
def add_noise(images):
    noisy_images = []
    for image in images:
        noisy_image = cv2.add(image, 0.018*np.random.randn(96,96,1))    # Adding random normal noise to the input image & clip the resulting noisy image between [-1,1]
        noisy_images.append(noisy_image.reshape(96,96,1))
    return noisy_images


random_noise_augmentation = True
if random_noise_augmentation:
    noisy_train_images = add_noise(clean_train_images)
    print("Shape of noisy_train_images:",np.shape(noisy_train_images))
print("************************************* Noisy Images with Key Points*********************************************")
if brightness_augmentation:
    fig = plt.figure(figsize=(20,8))
    for i in range(10):
        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        plot_sample(noisy_train_images[i], clean_train_keypoints[i], axis, "Brightness Augmentation: ")
    plt.show()
from skimage.util import random_noise
from skimage import data, img_as_float

im = array_to_img(clean_train_images[0])
original = img_as_float(im)

fig = plt.figure(figsize=(15,15))
sigmas = [0.01,0.1,0.25, 0.5]
for i in range(4):  
    noisy = random_noise(original, var=sigmas[i]**2) 
    #plt.subplot(1,4,i+1)
    axis = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    axis.imshow(noisy) 
    plt.title('gauss =' + str(sigmas[i]))
plt.show()
#Various methods under skimage to add noise
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

original = img_as_float(data.chelsea()[100:250, 50:300])

sigma = 0.155
noisy = random_noise(original, var=sigma**2)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print(f"Estimated Gaussian noise standard deviation = {sigma_est}")

ax[0, 0].imshow(noisy)
ax[0, 0].axis('off')
ax[0, 0].set_title('Noisy')
ax[0, 1].imshow(denoise_tv_chambolle(noisy, weight=0.1, multichannel=True))
ax[0, 1].axis('off')
ax[0, 1].set_title('TV')
ax[0, 2].imshow(denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15,
                multichannel=True))
ax[0, 2].axis('off')
ax[0, 2].set_title('Bilateral')
ax[0, 3].imshow(denoise_wavelet(noisy, multichannel=True, rescale_sigma=True))
ax[0, 3].axis('off')
ax[0, 3].set_title('Wavelet denoising')

ax[1, 1].imshow(denoise_tv_chambolle(noisy, weight=0.2, multichannel=True))
ax[1, 1].axis('off')
ax[1, 1].set_title('(more) TV')
ax[1, 2].imshow(denoise_bilateral(noisy, sigma_color=0.1, sigma_spatial=15,
                multichannel=True))
ax[1, 2].axis('off')
ax[1, 2].set_title('(more) Bilateral')
ax[1, 3].imshow(denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                rescale_sigma=True))
ax[1, 3].axis('off')
ax[1, 3].set_title('Wavelet denoising\nin YCbCr colorspace')
ax[1, 0].imshow(original)
ax[1, 0].axis('off')
ax[1, 0].set_title('Original')

fig.tight_layout()

plt.show()
#Writing a function for shift the image horizontal and verical
pixel_shifts = [12]

def shift_images(images, keypoints):
    shifted_images = []
    shifted_keypoints = []
    
    for shift in pixel_shifts:    # Augmenting over several pixel shift values
        for (shift_x,shift_y) in [(-shift,-shift),(-shift,shift),(shift,-shift),(shift,shift)]:
            M = np.float32([[1,0,shift_x],[0,1,shift_y]])
            
            for image, keypoint in zip(images, keypoints):
                shifted_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
                shifted_keypoint = np.array([(point+shift_x) if idx%2==0 else (point+shift_y) for idx, point in enumerate(keypoint)])
                
                if np.all(0.0<shifted_keypoint) and np.all(shifted_keypoint<96.0):
                    shifted_images.append(shifted_image.reshape(96,96,1))
                    shifted_keypoints.append(shifted_keypoint)
                    
    shifted_keypoints = np.clip(shifted_keypoints,0.0,96.0)
    
    return shifted_images, shifted_keypoints
shift_augmentation = True

if shift_augmentation:
    shifted_train_images, shifted_train_keypoints = shift_images(clean_train_images, clean_train_keypoints)
    print(f"Shape of shifted_train_images:",np.shape(shifted_train_images))
    print(f"Shape of shifted_train_keypoints:",np.shape(shifted_train_keypoints))
if shift_augmentation:
    print("*********************************************** Pixel Shifted Images *********************************************************")
    fig = plt.figure(figsize=(20,8))
    for i in range(10):
        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        plot_sample(shifted_train_images[i], shifted_train_keypoints[i], axis, "Shift Augmentation: ")
    plt.show()

if shift_augmentation:
    print("*********************************************** Pixel Shifted Images *********************************************************")
    fig = plt.figure(figsize=(20,8))
    for i in range(2141,2151,1):
        axis = fig.add_subplot(2, 5, i-2140, xticks=[], yticks=[])
        plot_sample(shifted_train_images[i], shifted_train_keypoints[i], axis, "Shift Augmentation: ")
    plt.show()