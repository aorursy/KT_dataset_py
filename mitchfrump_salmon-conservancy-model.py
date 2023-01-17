# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/train"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.image as mim

import resource



df = {"image":[],"species":[]}



# Doing this directly exceeds memory limits. Not 100% sure how to (a) measure this, (b) work around it.

# Could possibly build the dataframe one species at a time, then save them to CSV and merge them.

# However it would be nice to be able to use all the data in training...



for folder in check_output(["ls", "../input/train"]).decode("utf8").split('\n'):

    print(folder)

    contents = check_output(["ls", "../input/train/"+folder]).decode("utf8").split('\n')[:10]

    for image in contents:

        if image[-4:]!='.jpg':

#            print(resource.getrusage(resource.RUSAGE_SELF)[2]*resource.getpagesize()/1000000.0)

            continue

        df['image'].append(mim.imread("../input/train/"+folder+'/'+image))

        df['species'].append(folder)

    del contents
max0 = 0

max1 = 0



for x in df["image"]:

    sh = x.shape

    if sh[0]>max0:

        max0 = sh[0]

    if sh[1]>max1:

        max1 = sh[1]

   

print("The biggest image dimensions seen were:",max0,max1)



from scipy.stats import describe

avs = []

for x in df["image"]:

    avs.append(np.mean(x))

print("The average brightness among all images was:",np.mean(avs))
def normalize(image,newshape=(974,1732,3)):

    '''Takes in an image array of shape (x,y,3)

    @returns an image array of shape (974,1732,3) with average unraveled value 0

    by subtracting averages, and either extending with zeroes or cropping'''

    shape = image.shape

    if shape[0]>newshape[0]:

        image = image[:newshape[0],:,:]

    if shape[1]>newshape[1]:

        image = image[:,:newshape[1],:]

    image = image - np.mean(image,axis=None)

    newimage = np.zeros(newshape)

    newimage[:image.shape[0],:image.shape[1],:] = image

    return newimage



test = df["image"][0]

print(describe(np.reshape(test,(-1,3))))

res = normalize(test)

print(describe(np.reshape(res,(-1,3))))

    
import tensorflow as tf

from sklearn.model_selection import train_test_split



X = np.array([normalize(x) for x in df["image"]])

y = df["species"]

del df



from sklearn.preprocessing import LabelEncoder



enc = LabelEncoder()

enc.fit(y)

y = enc.transform(y)



train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(X,y,stratify=y)



image_h = 974

image_w = 1732

num_labels = 8

num_channels = 1 # grayscale



import numpy as np



def reformat(dataset, labels):

  dataset = dataset.reshape(

    (-1, image_h, image_w, num_channels)).astype(np.float32)

  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)

  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)

valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)

#test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)

print('Validation set', valid_dataset.shape, valid_labels.shape)

#print('Test set', test_dataset.shape, test_labels.shape)
#Sample code from here:

# http://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/



# import the necessary packages

from skimage.segmentation import slic

from skimage.segmentation import mark_boundaries

from skimage.util import img_as_float

from skimage import io

import matplotlib.pyplot as plt

import argparse

 

# construct the argument parser and parse the arguments

#ap = argparse.ArgumentParser()

#ap.add_argument("-i", "--image", required = True, help = "Path to the image")

#args = vars(ap.parse_args())

 

# load the image and convert it to a floating point data type

image = img_as_float(io.imread("../input/train/ALB/img_00003.jpg")) #img_as_float(io.imread(args["image"]))

 

# loop over the number of segments

for numSegments in (10, 30, 80):

	# apply SLIC and extract (approximately) the supplied number

	# of segments

	segments = slic(image, n_segments = numSegments, sigma = 5)

 

	# show the output of SLIC

	fig = plt.figure("Superpixels -- %d segments" % (numSegments))

	ax = fig.add_subplot(1, 1, 1)

	ax.imshow(mark_boundaries(image, segments))

	plt.axis("off")

 

# show the plots

plt.show()
#Sample code from here:

# http://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/

# The purpose of this segment is just to establish what kind of superpixels to look for



# import the necessary packages

from skimage.segmentation import slic

from skimage.segmentation import mark_boundaries

from skimage.util import img_as_float

from skimage import io

import matplotlib.pyplot as plt



 

# loop over the number of segments

for folder in check_output(["ls", "../input/train"]).decode("utf8").split('\n'):

    print(folder)

    contents = check_output(["ls", "../input/train/"+folder]).decode("utf8").split('\n')[:2]

    for imfile in contents:

        # load the image and convert it to a floating point data type

        if imfile[-4:]!='.jpg':

            continue

        image = img_as_float(io.imread("../input/train/"+folder+"/"+imfile))

        for numSegments in (10, 30, 50):

	        # apply SLIC and extract (approximately) the supplied number

	        # of segments

	        segments = slic(image, n_segments = numSegments, sigma = 5)

     

	        # show the output of SLIC

	        fig = plt.figure("Superpixels -- %d segments" % (numSegments))

	        ax = fig.add_subplot(1, 1, 1)

	        ax.imshow(mark_boundaries(image, segments))

	        plt.axis("off")

 

    # show the plots

    plt.show()
image_h = 974

image_w = 1732

num_labels = 8

num_channels = 1 # grayscale



import numpy as np



def reformat(dataset, labels):

  dataset = dataset.reshape(

    (-1, image_h, image_w, num_channels)).astype(np.float32)

  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)

  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)

valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)

#test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)

print('Validation set', valid_dataset.shape, valid_labels.shape)

#print('Test set', test_dataset.shape, test_labels.shape)