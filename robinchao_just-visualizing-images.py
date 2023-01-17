import os

import sys

import random



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from skimage.transform import resize

from skimage.morphology import label

from skimage.feature import hog

from skimage import exposure

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage.feature import canny

from skimage.filters import sobel

from skimage.morphology import watershed

from scipy import ndimage as ndi

import warnings

warnings.filterwarnings("ignore")

from skimage.segmentation import mark_boundaries

from scipy import signal

import cv2

import glob, pylab, pandas as pd

import pydicom, numpy as np

import tqdm

import gc

gc.enable()

import glob



from skimage.transform import resize

from skimage.morphology import label

from skimage import exposure
ROOT_FOLDER = '/kaggle/input/rsna-intracranial-hemorrhage-detection'

TRAIN_CSV = ROOT_FOLDER + '/stage_1_train.csv'

TRAIN_FOLDER = ROOT_FOLDER + '/stage_1_train_images'

TEST_FOLDER = ROOT_FOLDER + '/stage_1_test_images'
train_files = glob.glob(TRAIN_FOLDER + '/*.dcm')

len(train_files)


test_files = glob.glob(TEST_FOLDER + '/*.dcm')

len(test_files)
df = pd.read_csv(TRAIN_CSV,header=None)

df.head()
df.shape
import cv2

from IPython.display import display, Image

def cvshow(image, format='.png', rate=255 ):

    decoded_bytes = cv2.imencode(format, image*rate)[1].tobytes()

    display(Image(data=decoded_bytes))

    return
j = 0

nImg = 10

img_ar = np.empty(0)

while img_ar.shape[0]!=nImg:

    dcm_file = train_files[j]

    dcm_data = pydicom.read_file(dcm_file)

    img = np.expand_dims(dcm_data.pixel_array,axis=0)    

    if j==0:

        img_ar = img

    elif (j%100==0):

        print(j,'images loaded')

    else:

        img_ar = np.concatenate([img_ar,img],axis=0)

    j += 1
def imgtile(imgs,tile_w):

    assert imgs.shape[0]%tile_w==0,"'imgs' cannot divide by 'th'."

    r=imgs.reshape((-1,tile_w)+imgs.shape[1:])

    return np.hstack(np.hstack(r))



#usage

tiled = imgtile(img_ar,5)

# cvshow(tiled)

tiled.shape
img = tiled.astype(np.float32)

cvshow(cv2.resize( img, (1024,512), interpolation=cv2.INTER_LINEAR ))
plt.figure(figsize=(30,15))

plt.subplots_adjust(bottom=0.2, top=0.7, hspace=0)  #adjust this to change vertical and horiz. spacings..

nImg = 3  #no. of images to process

for j in range(nImg):

    q = j+1

    img = np.array(pydicom.read_file(train_files[j]).pixel_array)

    

#     # Contrast stretching

    p2, p97 = np.percentile(img, (2, 97))

    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p97))

    

    # Equalization

    img_eq = exposure.equalize_hist(img)



    # Adaptive Equalization

    img_adapteq = exposure.equalize_adapthist(img)

    

    plt.subplot(nImg,7,q*7-6)

    plt.imshow(img, cmap=plt.cm.bone)

    plt.title('Original Image')

    

    

    plt.subplot(nImg,7,q*7-5)    

    plt.imshow(img_rescale, cmap=plt.cm.bone)

    plt.title('Contrast stretching')

    

    

    plt.subplot(nImg,7,q*7-4)

    plt.imshow(img_eq, cmap=plt.cm.bone)

    plt.title('Equalization')

    

    

    plt.subplot(nImg,7,q*7-3)

    plt.imshow(img_adapteq, cmap=plt.cm.bone)

    plt.title('Adaptive Equalization')

plt.show()
plt.figure(figsize=(30,15))

plt.subplots_adjust(bottom=0.2, top=0.7, hspace=0)  #adjust this to change vertical and horiz. spacings..

nImg = 3  #no. of images to process

for j in range(nImg):

    q = j+1

    img = np.array(pydicom.read_file(test_files[j]).pixel_array)

    

#     # Contrast stretching

    p2, p97 = np.percentile(img, (2, 97))

    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p97))

    

    # Equalization

    img_eq = exposure.equalize_hist(img)



    # Adaptive Equalization

    img_adapteq = exposure.equalize_adapthist(img)

    

    plt.subplot(nImg,7,q*7-6)

    plt.imshow(img, cmap=plt.cm.bone)

    plt.title('Original Image')

    

    

    plt.subplot(nImg,7,q*7-5)    

    plt.imshow(img_rescale, cmap=plt.cm.bone)

    plt.title('Contrast stretching')

    

    

    plt.subplot(nImg,7,q*7-4)

    plt.imshow(img_eq, cmap=plt.cm.bone)

    plt.title('Equalization')

    

    

    plt.subplot(nImg,7,q*7-3)

    plt.imshow(img_adapteq, cmap=plt.cm.bone)

    plt.title('Adaptive Equalization')

plt.show()