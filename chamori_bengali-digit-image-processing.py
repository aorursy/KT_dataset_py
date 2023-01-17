import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.ndimage 

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import glob

import zipfile
import skimage
18# Create a new list to hold the names of the first 25 jpg images

filelist = ['BDRW_train_1/digit_0.jpg',

 'BDRW_train_1/digit_1.jpg',

 'BDRW_train_1/digit_10.jpg',

 'BDRW_train_1/digit_100.jpg',

 'BDRW_train_1/digit_1000.jpg',

 'BDRW_train_1/digit_1001.jpg',

 'BDRW_train_1/digit_1002.jpg',

 'BDRW_train_1/digit_1003.jpg',

 'BDRW_train_1/digit_1005.jpg',

 'BDRW_train_1/digit_1006.jpg',

 'BDRW_train_1/digit_1007.jpg',

 'BDRW_train_1/digit_1008.jpg',

 'BDRW_train_1/digit_1009.jpg',

 'BDRW_train_1/digit_101.jpg',

 'BDRW_train_1/digit_1011.jpg',

 'BDRW_train_1/digit_1012.jpg',

 'BDRW_train_1/digit_1013.jpg',

 'BDRW_train_1/digit_1014.jpg',

 'BDRW_train_1/digit_1015.jpg',

 'BDRW_train_1/digit_1016.jpg',

 'BDRW_train_1/digit_1017.jpg',

 'BDRW_train_1/digit_1018.jpg',

 'BDRW_train_1/digit_102.jpg',

 'BDRW_train_1/digit_1020.jpg',

 'BDRW_train_1/digit_1021.jpg']
z = zipfile.ZipFile('../input/BDRW_train/BDRW_train_1.zip', "r")

for name in z.namelist():

    if name in filelist:

        z.extract(name)
train = [f for f in glob.glob("BDRW_train_1/*")]
i = 0

plt.figure(figsize=(7.5,7.5))

for k in train:

    img = mpimg.imread(k)

    plt.subplot(5,5,i+1); plt.imshow(img[:,:,0],cmap=plt.cm.inferno_r); plt.axis('off')

    plt.title('ID = ' + str(i), fontsize=7)

    i += 1

plt.tight_layout()
# Try different Color schemes 

# Make sure we have our particular Bengali digit of interest by ensuring that we always call the right index

for index,name in enumerate(train):

    if name == 'BDRW_train_1/digit_1011.jpg':

        plt.figure(figsize=(4.5,4.5))

        plt.subplot(441)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.flag)

        plt.title('flag', fontsize=8)

        plt.axis('off')

        

        plt.subplot(442)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.gnuplot_r)

        plt.title('Gnuplot', fontsize=8)

        plt.axis('off')

        

        plt.subplot(443)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.Blues)

        plt.title('Blues', fontsize=8)

        plt.axis('off')

        

        plt.subplot(444)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.prism_r)

        plt.title('prism_r', fontsize=8)

        plt.axis('off')

        

        plt.subplot(445)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.cubehelix_r)

        plt.title('Cubehelix', fontsize=8)

        plt.axis('off')

        

        plt.subplot(446)

        plt.imshow(mpimg.imread(train[index])[:,:,:])

        plt.title('Original', fontsize=8)

        plt.axis('off')

        

        plt.subplot(447)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.nipy_spectral_r)

        plt.title('Nipy_spectral_r', fontsize=8)

        plt.axis('off')

        

        plt.subplot(448)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.Accent)

        plt.title('Accent', fontsize=8)

        plt.axis('off')

        

        plt.subplot(449)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.copper)

        plt.title('copper', fontsize=8)

        plt.axis('off')

        

        plt.subplot(4,4,10)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.winter_r)

        plt.title('winter_r', fontsize=8)

        plt.axis('off')

        

        plt.subplot(4,4,11)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.Paired_r)

        plt.title('Paired_r', fontsize=8)

        plt.axis('off')

        

        plt.subplot(4,4,12)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.coolwarm)

        plt.title('coolwarm', fontsize=8)

        plt.axis('off')

        

        plt.subplot(4,4,13)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.flag_r)

        plt.title('flag_r', fontsize=8)

        plt.axis('off')

        

        plt.subplot(4,4,14)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.Set1)

        plt.title('Set1', fontsize=8)

        plt.axis('off')

        

        plt.subplot(4,4,15)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.afmhot_r)

        plt.title('afmhot_r', fontsize=8)

        plt.axis('off')

        

        plt.subplot(4,4,16)

        plt.imshow(mpimg.imread(train[index])[:,:,0],cmap=plt.cm.prism)

        plt.title('prism', fontsize=8)

        plt.axis('off')
for index,name in enumerate(train):

    if name == 'BDRW_train_1/digit_1011.jpg':

        plt.figure(figsize=(6,6))

        

        plt.subplot(131)

        plt.imshow(mpimg.imread(train[index])[:,:,0], cmap=plt.cm.bone_r)

        plt.title('No Filter', fontsize=10)

        

        plt.subplot(132)

        plt.imshow(scipy.ndimage.uniform_filter(mpimg.imread(train[index])[:,:,0]),cmap=plt.cm.bone_r)

        plt.title('Uniform Filter', fontsize=10)

        

        plt.subplot(133)

        plt.imshow(scipy.ndimage.gaussian_filter(mpimg.imread(train[index])[:,:,0],2.5),cmap=plt.cm.bone_r)

        plt.title('Gaussian Blurring', fontsize=10)

        plt.tight_layout()
for index,name in enumerate(train):

    if name == 'BDRW_train_1/digit_1011.jpg':

        plt.figure(figsize=(6,6))

        

        plt.subplot(131)

        plt.imshow(mpimg.imread(train[index])[:,:,0], cmap=plt.cm.cubehelix_r)

        plt.title('No Filter', fontsize=10)

        

        plt.subplot(132)

        plt.imshow(scipy.ndimage.maximum_filter(mpimg.imread(train[index])[:,:,0],3.5), cmap=plt.cm.cubehelix_r)

        plt.title('Maximum Filter', fontsize=10)

        

        plt.subplot(133)

        img = mpimg.imread(train[index])[:,:,0]

        alpha = 30

        sharpened = img + alpha * (img - scipy.ndimage.gaussian_filter(mpimg.imread(train[index])[:,:,0],1))

        plt.imshow(sharpened,cmap=plt.cm.cubehelix_r )

        plt.title('Laplacian Filter', fontsize=10)

        plt.tight_layout()
for index,name in enumerate(train):

    if name == 'BDRW_train_1/digit_1011.jpg':

        plt.figure(figsize=(6,6))

        

        plt.subplot(131)

        plt.imshow(mpimg.imread(train[index])[:,:,0], cmap=plt.cm.flag)

        

        plt.subplot(132)

        plt.imshow(scipy.ndimage.median_filter(mpimg.imread(train[index])[:,:,0],3),cmap=plt.cm.flag)

        

        plt.subplot(133)

        plt.imshow(skimage.filter.denoise_bilateral(mpimg.imread(train[index])[:,:,:]),cmap=plt.cm.flag)

        plt.tight_layout()
mpimg.imread(train[index])[:,:,0] < 65
for index,name in enumerate(train):

    if name == 'BDRW_train_1/digit_1011.jpg':

        plt.figure(figsize=(7,7))

        

        plt.subplot(141)

        img = mpimg.imread(train[index])[:,:,0] > 60

        plt.imshow(img, cmap=plt.cm.inferno_r)

        plt.axis('off')

        

        plt.subplot(142)

        img = mpimg.imread(train[index])[:,:,0] > 75

        plt.imshow(img,cmap=plt.cm.inferno_r)

        plt.axis('off')

        

        plt.subplot(143)

        img = mpimg.imread(train[index])[:,:,0] > 100

        plt.imshow(img,cmap=plt.cm.inferno_r)

        plt.axis('off')

        

        plt.subplot(144)

        img = mpimg.imread(train[index])[:,:,0] > 250

        plt.imshow(img,cmap=plt.cm.inferno_r)

        plt.axis('off')

        plt.tight_layout()
# Import the relevant modules to call Otsu THresholding

from skimage import data

from skimage import filters
img = mpimg.imread(train[index])[:,:,:]

plt.figure(figsize=(4,4))

val = filters.threshold_otsu(img)

mask = img < val



plt.subplot(121)

plt.imshow(img)



plt.subplot(122)

plt.imshow(mask)

plt.tight_layout()