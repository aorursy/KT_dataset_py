# importing all the required libraries

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import skimage.io as io

from skimage.transform import rotate, AffineTransform, warp

from skimage.util import random_noise

from skimage.filters import gaussian

import matplotlib.pyplot as plt

import PIL.Image

import matplotlib.pyplot as plt

import torch

from torchvision import transforms




def imshow(img, transform):

    """helper function to show data augmentation

    :param img: path of the image

    :param transform: data augmentation technique to apply"""

    

    img = PIL.Image.open(img)

    fig, ax = plt.subplots(1, 2, figsize=(15, 4))

    ax[0].set_title(f'original image {img.size}')

    ax[0].imshow(img)

    img = transform(img)

    ax[1].set_title(f'transformed image {img.size}')

    ax[1].imshow(img)

loader_transform = transforms.Resize((140, 140))



imshow('../input/lion-image/lion.jpg', loader_transform)
loader_transform = transforms.CenterCrop(140)

imshow('../input/lion-image/lion.jpg', loader_transform)
# horizontal flip with probability 1 (default is 0.5)

loader_transform = transforms.RandomHorizontalFlip(p=1)

imshow('../input/lion-image/lion.jpg', loader_transform)
# left, top, right, bottom

loader_transform = transforms.Pad((2, 5, 0, 5))

imshow('../input/lion-image/lion.jpg', loader_transform)
loader_transform = transforms.RandomRotation(30)

imshow('../input/lion-image/lion.jpg', loader_transform)
# random affine transformation of the image keeping center invariant

loader_transform = transforms.RandomAffine(0, translate=(0.4, 0.5))

imshow('../input/lion-image/lion.jpg', loader_transform)
# reading the image using its path

image = io.imread('../input/lion-image/lion.jpg')



# shape of the image

print(image.shape)



# displaying the image

io.imshow(image)
print('Rotated Image')

#rotating the image by 45 degrees

rotated = rotate(image, angle=45, mode = 'wrap')

#plot the rotated image

io.imshow(rotated)
#apply shift operation

transform = AffineTransform(translation=(25,25))

wrapShift = warp(image,transform,mode='wrap')

plt.imshow(wrapShift)

plt.title('Wrap Shift')
#flip image left-to-right

flipLR = np.fliplr(image)



plt.imshow(flipLR)

plt.title('Left to Right Flipped')
#flip image up-to-down

flipUD = np.flipud(image)



plt.imshow(flipUD)

plt.title('Up Down Flipped')
#standard deviation for noise to be added in the image

sigma=0.155

#add random noise to the image

noisyRandom = random_noise(image,var=sigma**2)



plt.imshow(noisyRandom)

plt.title('Random Noise')
#blur the image

blurred = gaussian(image,sigma=1,multichannel=True)



plt.imshow(blurred)

plt.title('Blurred Image')
img = PIL.Image.open('../input/lion-image/lion.jpg')

fig, ax = plt.subplots(2, 2, figsize=(16, 10))



# brightness

loader_transform1 = transforms.ColorJitter(brightness=2)

img1 = loader_transform1(img)

ax[0, 0].set_title(f'brightness')

ax[0, 0].imshow(img1)



# contrast

loader_transform2 = transforms.ColorJitter(contrast=2)

img2 = loader_transform2(img)

ax[0, 1].set_title(f'contrast')

ax[0, 1].imshow(img2)



# saturation

loader_transform3 = transforms.ColorJitter(saturation=2)

img3 = loader_transform3(img)

ax[1, 0].set_title(f'saturation')

ax[1, 0].imshow(img3)

fig.savefig('color augmentation', bbox_inches='tight')



# hue

loader_transform4 = transforms.ColorJitter(hue=0.2)

img4 = loader_transform4(img)

ax[1, 1].set_title(f'hue')

ax[1, 1].imshow(img4)



fig.savefig('color augmentation', bbox_inches='tight')
