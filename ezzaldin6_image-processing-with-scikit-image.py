import numpy as np 
import pandas as pd 
from skimage.data import rocket, camera, astronaut, clock, coffee, page, coins, gravel, hubble_deep_field, logo
from skimage import color
from skimage.filters import try_all_threshold, threshold_otsu, threshold_local, sobel, gaussian
from skimage.exposure import equalize_hist, equalize_adapthist
from skimage.transform import rotate, resize, rescale
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import os
%matplotlib inline
# representing an Image as an Multidimentional array
rocket_image=rocket()
rocket_image
#function to visualize the images
def show_image(img, title, cmap_type='gray'):
    plt.imshow(img, cmap=cmap_type)
    plt.title(title)
    plt.show()
show_image(rocket_image, 'rocket')
# what is the shape of the array that represents the rocket image
print('height= ',rocket_image.shape[0])
print('width= ', rocket_image.shape[1])
print('number of channels= ', rocket_image.shape[2])
gray_rocket_img=color.rgb2gray(rocket_image)
show_image(gray_rocket_img, 'gray rocket')
print(gray_rocket_img.shape)
as_image=astronaut()
red_as=as_image[:,:,0]
green_as=as_image[:,:,1]
blue_as=as_image[:,:,2]
fig=plt.figure(figsize=(15,5))
for i,j,k in zip([as_image, red_as, green_as, blue_as], range(4), ['original','red', 'green', 'blue']):
    ax=fig.add_subplot(1,4,j+1)
    plt.imshow(i)
    plt.title(k)
plt.show()
fig=plt.figure(figsize=(15,3))
for i,j,k in zip([red_as, green_as, blue_as], range(3), ['red', 'green', 'blue']):
    ax=fig.add_subplot(1,3,j+1)
    plt.hist(i.ravel(), color=k, bins=256)
    plt.title(k)
plt.show()
# Flipping images: Vertically
vertical_as_image=np.flipud(as_image)
show_image(vertical_as_image, 'vertical Flipping')
# Horizontally
horizontal_as_image=np.fliplr(as_image)
show_image(horizontal_as_image, 'horizontal Flipping')
show_image(coffee(), 'coffee')
# Optimal Threshold
coffee_image=coffee()
#convert it to grayscale image
gray_coffee_img=color.rgb2gray(coffee_image)
thresh=threshold_otsu(gray_coffee_img)
thresh_img=gray_coffee_img>thresh
show_image(thresh_img, 'effect of optimal thresholding')
page_image=page()
local_thresh=threshold_local(page_image,51)
local_thresh_img=page_image>local_thresh
show_image(local_thresh_img, 'effect of local thresholding')
coins_image=coins()
sobel_coins_img=sobel(coins_image)
fig=plt.figure(figsize=(10,5))
for i,j,k in zip([coins_image, sobel_coins_img], range(2), ['original', 'sobel filter']):
    ax=fig.add_subplot(1,2,j+1)
    ax.imshow(i, cmap='gray')
    ax.set_title(k)
plt.show()
# let's see sobel filter for coffee image
sobel_coffee_img=sobel(gray_coffee_img)
fig=plt.figure(figsize=(10,5))
for i,j,k in zip([coffee_image, sobel_coffee_img], range(2), ['original', 'sobel filter']):
    ax=fig.add_subplot(1,2,j+1)
    ax.imshow(i, cmap='gray')
    ax.set_title(k)
plt.show()
gaussian_coffee_img=gaussian(coffee_image, multichannel=True)
fig=plt.figure(figsize=(10,5))
for i,j,k in zip([coffee_image, gaussian_coffee_img], range(2), ['original', 'gaussian filter']):
    ax=fig.add_subplot(1,2,j+1)
    ax.imshow(i, cmap='gray')
    ax.set_title(k)
plt.show()
he_coins_img=equalize_hist(coins_image)
ahe_coins_img=equalize_adapthist(coins_image)
fig=plt.figure(figsize=(15,5))
for i,j,k in zip([coins_image, he_coins_img, ahe_coins_img], range(3), ['original', 'histogram equalization', 'Adaptive histogram eqalization']):
    ax=fig.add_subplot(1,3,j+1)
    ax.imshow(i, cmap='gray')
    ax.set_title(k)
plt.show()
# Rotating Clockwise
rotated_rocket_1=rotate(rocket_image,-90)
rotated_rocket_2=rotate(rocket_image,-150)
rotated_rocket_3=rotate(rocket_image,-270)
fig=plt.figure(figsize=(20,5))
for i,j,k in zip([rocket_image, rotated_rocket_1, rotated_rocket_2, rotated_rocket_3], range(4), ['original', 'cw 90', 'cw 150', 'cw 270']):
    ax=fig.add_subplot(1,4,j+1)
    ax.imshow(i)
    ax.set_title(k)
plt.show()
# rotating anticlockwise
rotated_rocket_4=rotate(rocket_image,90)
rotated_rocket_5=rotate(rocket_image,150)
rotated_rocket_6=rotate(rocket_image,270)
fig=plt.figure(figsize=(20,5))
for i,j,k in zip([rocket_image, rotated_rocket_4, rotated_rocket_5, rotated_rocket_6], range(4), ['original', 'acw 90', 'acw 150', 'acw 270']):
    ax=fig.add_subplot(1,4,j+1)
    ax.imshow(i)
    ax.set_title(k)
plt.show()
scikit_image_logo=logo()
rescaled_logo=rescale(scikit_image_logo, 1/2, anti_aliasing=True, multichannel=True)
fig=plt.figure(figsize=(10,5))
for i,j,k in zip([scikit_image_logo, rescaled_logo], range(2), ['original', 'rescaled-logo']):
    ax=fig.add_subplot(1,2,j+1)
    ax.imshow(i, cmap='gray')
    ax.set_title(k)
plt.show()
new_height=scikit_image_logo.shape[0]*3
new_width=scikit_image_logo.shape[1]*4
resized_logo=resize(scikit_image_logo, (new_height, new_width), anti_aliasing=True)
fig=plt.figure(figsize=(15,5))
for i,j,k in zip([scikit_image_logo, resized_logo], range(2), ['original', 'resized-logo']):
    ax=fig.add_subplot(1,2,j+1)
    ax.imshow(i, cmap='gray')
    ax.set_title(k)
plt.show()
# Apply noise in scikit-image
noisy_as_image=random_noise(as_image)
fig=plt.figure(figsize=(10,5))
for i,j,k in zip([as_image, noisy_as_image], range(2), ['original', 'noisy_image']):
    ax=fig.add_subplot(1,2,j+1)
    ax.imshow(i, cmap='gray')
    ax.set_title(k)
plt.show()
TV_as_img=denoise_tv_chambolle(noisy_as_image, multichannel=True)
bilateral_as_img=denoise_bilateral(noisy_as_image, multichannel=True)
fig=plt.figure(figsize=(20,5))
for i,j,k in zip([as_image, noisy_as_image, TV_as_img, bilateral_as_img], range(4), ['original', 'noisy image', 'TV', 'Bilateral']):
    ax=fig.add_subplot(1,4,j+1)
    ax.imshow(i, cmap='gray')
    ax.set_title(k)
plt.show()
domino_image=plt.imread('data:image/png;base64,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')
gray_domino_img=color.rgb2gray(domino_image)
# preparing the image
thresh=threshold_otsu(gray_domino_img)
thresh_domino_img=gray_domino_img>thresh
fig=plt.figure(figsize=(15,5))
for i,j,k in zip([domino_image, gray_domino_img, thresh_domino_img], range(3), ['original', 'gray_image', 'otsu_filter']):
    ax=fig.add_subplot(1,3,j+1)
    ax.imshow(i, cmap='gray')
    ax.set_title(k)
plt.show()
# use find_contours()
contours_domino_img=find_contours(thresh_domino_img, 0.8)
for contour in contours_domino_img:
    print(contour.shape)