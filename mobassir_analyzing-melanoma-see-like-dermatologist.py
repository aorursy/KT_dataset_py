!pip install imutils
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

from __future__ import unicode_literals

from __future__ import print_function

from __future__ import division

from __future__ import absolute_import

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from pathlib import Path

import pandas as pd

from torch.utils.data import Dataset,DataLoader



from scipy.spatial import distance as dist

from imutils import perspective

from imutils import contours

import matplotlib.pyplot as plt

import numpy as np

import imutils

import cv2



import skimage.measure

import imageio

from PIL import Image

import requests

from io import BytesIO

from torchvision import transforms as T

import torch.nn as nn

import torch

import torch.nn.functional as F

from sklearn.model_selection import GroupKFold

from kaggle_datasets import KaggleDatasets



from scipy.spatial.distance import euclidean

from imutils import perspective

from imutils import contours

import numpy as np

import imutils

import cv2

import matplotlib.pyplot as plt



from glob import glob

import pandas as pd

from sklearn.model_selection import GroupKFold

import cv2

from skimage import io

import albumentations as A

import torch

import os

from datetime import datetime

import time

import random

import cv2

import pandas as pd

import numpy as np

import albumentations as A

import matplotlib.pyplot as plt

from albumentations.pytorch.transforms import ToTensorV2

from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset,DataLoader

from torch.utils.data.sampler import SequentialSampler, RandomSampler

from torch.nn import functional as F

from glob import glob

import sklearn

from torch import nn





import keras

import numpy as np

import tensorflow as tf

from keras.models import model_from_json, load_model

import json



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

import tensorflow as tf

from functools import partial



import glob

import numpy as np

import cv2

from skimage import filters as skifilters

from scipy import ndimage

from skimage import filters

import matplotlib.pyplot as plt

import tqdm

from sklearn.utils import shuffle

import pandas as pd



import os

import h5py

import time

import json

import warnings

from PIL import Image



from fastprogress.fastprogress import master_bar, progress_bar

from sklearn.metrics import accuracy_score, roc_auc_score

from torchvision import models

import pdb

import albumentations as A

from albumentations.pytorch.transforms import ToTensor

import matplotlib.pyplot as plt



import pickle 

import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def list_files(path:Path):

    return [o for o in path.iterdir()]
path = Path('../input/jpeg-melanoma-768x768/')

df_path = Path('../input/jpeg-melanoma-768x768/')

im_sz = 256

bs = 16
train_fnames = list_files(path/'train')

df = pd.read_csv(df_path/'train.csv')

df.head()




df.target.value_counts(),df.shape





GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-768x768')
def decode_image(image):

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    return image
def read_tfrecord(example, labeled):

    tfrecord_format = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "target": tf.io.FixedLenFeature([], tf.int64)

    } if labeled else {

        "image": tf.io.FixedLenFeature([], tf.string),

        "image_name": tf.io.FixedLenFeature([], tf.string)

    }

    example = tf.io.parse_single_example(example, tfrecord_format)

    image = decode_image(example['image'])

    if labeled:

        label = tf.cast(example['target'], tf.int32)

        return image, label

    idnum = example['image_name']

    return image, idnum
def load_dataset(filenames, labeled=True, ordered=False):

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset


BATCH_SIZE = 8

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGE_SIZE = [768, 768]

TRAINING_FILENAMES, VALID_FILENAMES = train_test_split(

    tf.io.gfile.glob(GCS_PATH + '/train*.tfrec'),

    test_size=0.2, random_state=5

)

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')

print('Train TFRecord Files:', len(TRAINING_FILENAMES))

print('Validation TFRecord Files:', len(VALID_FILENAMES))

print('Test TFRecord Files:', len(TEST_FILENAMES))




def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    #dataset = dataset.map(augmentation_pipeline, num_parallel_calls=AUTOTUNE)

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset



train_dataset = get_training_dataset()
def show_batch(image_batch, label_batch):

    plt.figure(figsize=(15,15))

    for n in range(8):

        ax = plt.subplot(8,8,n+1)

        plt.imshow(image_batch[n])

        if label_batch[n]:

            plt.title("MALIGNANT(1)")

        else:

            plt.title("BENIGN(0)")

        plt.axis("off")
%%time

for i in range(0,10):

    image_batch, label_batch = next(iter(train_dataset))

    for j in range(0,8):

        var = label_batch[j].numpy()

        if(var!=0):

            show_batch(image_batch.numpy(), label_batch.numpy())



print("Samples with Melanoma")

imgs = df[df.target==1]['image_name'].values

_, axs = plt.subplots(2, 3, figsize=(20, 8))

axs = axs.flatten()

for f_name,ax in zip(imgs[10:20],axs):

    img = Image.open(path/f'train/{f_name}.jpg')

    ax.imshow(img)

    ax.axis('off')    

plt.show()


# Usage: This script will measure different objects in the frame using a reference object 







# Function to show array of images (intermediate results)



def show_images(images):

    for i, img in enumerate(images):

        plt.figure(figsize=(20,20))

        plt.imshow(img)

        plt.show()

       



        

imgs = df[df.target==1]['image_name'].values

print("Samples with Melanoma")

for f_name,ax in zip(imgs[:100],axs):

 

    

    im1 = Image.open(path/f'train/{f_name}.jpg')

    print(path/f'train/{f_name}.jpg')

    im1.save('./a.png')

    img_path = '../working/a.png'







    '''load our image from disk, convert it to grayscale, and then smooth it using a Gaussian filter.

    We then perform edge detection along with a dilation + erosion to close any gaps 

    in between edges in the edge map

    '''



    # Read image and preprocess

    image = cv2.imread(img_path)

 

    #image = img



    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)



    edged = cv2.Canny(blur, 50, 100)

    edged = cv2.dilate(edged, None, iterations=1)

    edged = cv2.erode(edged, None, iterations=1)



    #show_images([blur, edged])



    '''find contours (i.e., the outlines) that correspond to the objects in our edge map.'''

    # Find contours

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)



    # Sort contours from left to right as leftmost contour is reference object

    try:

        '''These contours are then sorted from left-to-right (allowing us to extract our reference object)'''

        (cnts, _) = contours.sort_contours(cnts)

         # Remove contours which are not large enough

        for k in range(0,20):

            try:

                cnts = [x for x in cnts if cv2.contourArea(x) > k]

                # Reference object dimensions

                # Here for reference I have used a 2cm x 2cm square

                mid = len(cnts)//2

                ref_object = cnts[mid]

            except:

                pass

    except:

        #print("An exception occurred") 

        continue



    #print(len(cnts))

    #print(cnts)

    #cv2.drawContours(image, cnts, -1, (0,255,0), 3)



    #show_images([image, edged])

    #print(len(cnts))



    # compute the rotated bounding box of the contour

    orig = image.copy()

    box = cv2.minAreaRect(ref_object)

    box = cv2.boxPoints(box)

    box = np.array(box, dtype="int")

    

    # order the points in the contour such that they appear

    # in top-left, top-right, bottom-right, and bottom-left

    # order, then draw the outline of the rotated bounding

    # box

    

    box = perspective.order_points(box)

    

    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # loop over the original points and draw them

    for (x, y) in box:

        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        

    (tl, tr, br, bl) = box

    dist_in_pixel = euclidean(tl, tr)

    dist_in_cm = 2

    pixel_per_cm = dist_in_pixel/dist_in_cm

    largestht = []

    largestwid = []

    # Draw remaining contours

    for cnt in cnts:

        box = cv2.minAreaRect(cnt)

        box = cv2.boxPoints(box)

        box = np.array(box, dtype="int")

        box = perspective.order_points(box)

        (tl, tr, br, bl) = box

        cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)

        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))

        mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))

        wid = euclidean(tl, tr)/pixel_per_cm

        ht = euclidean(tr, br)/pixel_per_cm

        largestht.append(ht)

        largestwid.append(wid)

       

        #cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 

        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        #cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), 

        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    show_images([image])   

    if(len(largestht)>0):

        a = largestht.index(max(largestht))

        b = largestwid.index(max(largestwid))

        largestht1 = largestht[b]

        largestwid1 = largestwid[b]

        largestht = largestht[a]

        largestwid = largestwid[a]

        



        print("Rectangle 1  has : HEIGHT = ",largestht,"and WIDTH = ",largestwid)

        print("Rectangle 2  has : HEIGHT = ",largestht1,"and WIDTH = ",largestwid1)

    
def shades_gray(image, njet=0, mink_norm=1, sigma=1):

    """

    Estimates the light source of an input_image as proposed in:

    J. van de Weijer, Th. Gevers, A. Gijsenij

    "Edge-Based Color Constancy"

    IEEE Trans. Image Processing, accepted 2007.

    Depending on the parameters the estimation is equal to Grey-World, Max-RGB, general Grey-World,

    Shades-of-Grey or Grey-Edge algorithm.

    :param image: rgb input image (NxMx3)

    :param njet: the order of differentiation (range from 0-2)

    :param mink_norm: minkowski norm used (if mink_norm==-1 then the max

           operation is applied which is equal to minkowski_norm=infinity).

    :param sigma: sigma used for gaussian pre-processing of input image

    :return: illuminant color estimation

    :raise: ValueError

    

    Ref: https://github.com/MinaSGorgy/Color-Constancy

    """

    gauss_image = filters.gaussian(image, sigma=sigma, multichannel=True)

    if njet == 0:

        deriv_image = [gauss_image[:, :, channel] for channel in range(3)]

    else:   

        if njet == 1:

            deriv_filter = filters.sobel

        elif njet == 2:

            deriv_filter = filters.laplace

        else:

            raise ValueError("njet should be in range[0-2]! Given value is: " + str(njet))     

        deriv_image = [np.abs(deriv_filter(gauss_image[:, :, channel])) for channel in range(3)]

    for channel in range(3):

        deriv_image[channel][image[:, :, channel] >= 255] = 0.

    if mink_norm == -1:  

        estimating_func = np.max 

    else:

        estimating_func = lambda x: np.power(np.sum(np.power(x, mink_norm)), 1 / mink_norm)

    illum = [estimating_func(channel) for channel in deriv_image]

    som   = np.sqrt(np.sum(np.power(illum, 2)))

    illum = np.divide(illum, som)

    return illum





def correct_image(image, illum):

    """

    Corrects image colors by performing diagonal transformation according to 

    given estimated illumination of the image.

    :param image: rgb input image (NxMx3)

    :param illum: estimated illumination of the image

    :return: corrected image

    

    Ref: https://github.com/MinaSGorgy/Color-Constancy

    """

    correcting_illum = illum * np.sqrt(3)

    corrected_image = image / 255.

    for channel in range(3):

        corrected_image[:, :, channel] /= correcting_illum[channel]

    return np.clip(corrected_image, 0., 1.)




# Color Transformations

mx    = correct_image(image, shades_gray(image, njet=0, mink_norm=-1, sigma=0))  # MaxRGB Constancy

gw    = correct_image(image, shades_gray(image, njet=0, mink_norm=+1, sigma=0))  # Gray World Constancy 

hsv   = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)                                   # HSV Color Space

lab   = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)                                   # CIELab Color Space



# Concatenate to Output Image

op    = np.concatenate((gw/255,np.expand_dims(hsv[:,:,0]/179,axis=2),hsv[:,:,1:]/255,

                        np.expand_dims(lab[:,:,0]/255,axis=2),lab[:,:,1:]/128),axis=2)
plt.imshow(op[:,:,:3]*255)


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image

        center = [int(w/2), int(h/2)]

    if radius is None: # use the smallest distance between the center and image walls

        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]

    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius

    return mask
circa_mask            = create_circular_mask(op.shape[0], op.shape[1], radius=200).astype(bool)

op                    = np.multiply(op, np.dstack((circa_mask,circa_mask,circa_mask,circa_mask,circa_mask,

                                                             circa_mask,circa_mask,circa_mask,circa_mask)))        
img1 = op[:,:,:3]*255
img1.shape


imageio.imwrite('filename1.png', img1)
a = cv2.imread('../working/filename1.png')

plt.imshow(a)
imo = Image.fromarray((gw*255).astype(np.uint8))

imo
#os.listdir('../input/jpeg-melanoma-768x768/train')


def midpoint(ptA, ptB):

    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)





def show_image(images):

    plt.figure(figsize=(20,20))

    plt.imshow(images)

    plt.show()

       





path = '../input/jpeg-melanoma-768x768/train/ISIC_4789377.jpg' #"../working/filename1.png"

im1 = Image.open(path)

im1.save('./c.png')

path = '../working/c.png'



    

width = 0.99





image = cv2.imread(path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (7, 7), 0)



edged = cv2.Canny(gray, 50, 100)

show_image(edged)

edged = cv2.dilate(edged, None, iterations=1)

edged = cv2.erode(edged, None, iterations=1)

show_image( edged)









cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)

print("Total number of contours are: ", len(cnts))

if(len(cnts)>0):

    (cnts, _) = contours.sort_contours(cnts)

pixelPerMetric = None

count = 0

totals = []

total = len(cnts)

for c in cnts:

    if cv2.contourArea(c) < 500:

        continue

    count += 1



    orig = image.copy()

    box = cv2.minAreaRect(c)

    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)

    box = np.array(box, dtype="int")



    box = perspective.order_points(box)

    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)



    for (x, y) in box:

        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)





    (tl, tr, br, bl) = box

    (tltrX, tltrY) = midpoint(tl, tr)

    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)

    (trbrX, trbrY) = midpoint(tr, br)



    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)

    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)

    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)

    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)



    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)

    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)



    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))

    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))



    if pixelPerMetric is None:

        pixelPerMetric = dB / width



    dimA = dA / pixelPerMetric

    dimB = dB / pixelPerMetric



    cv2.putText(orig, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.putText(orig, "{:.1f}in".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    totals.append(orig)

    plt.imshow(orig)

print("Total contours processed: ", count)

    
n_row = 2

n_col = 2

_, axs = plt.subplots(n_row, n_col, figsize=(12, 12))

axs = axs.flatten()



for i in range(len(totals)):

    axs[i].imshow(totals[i])

plt.show()


  

# cv2.cvtColor is applied over the 

# image input with applied parameters 

# to convert the image in grayscale  

image = cv2.imread('../input/jpeg-melanoma-768x768/train/ISIC_0232101.jpg')

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

  

# applying different thresholding  

# techniques on the input image 

# all pixels value above 120 will  

# be set to 255 



ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO) 



  

# the window showing output images 

# with the corresponding thresholding  

# techniques applied to the input images 

plt.imshow(thresh) 

df_meta = pd.read_csv('../input/melanoma-merged-external-data-512x512-jpeg/folds_13062020.csv')
df_meta.target.value_counts()


def get_train_transforms():

    return A.Compose([

            A.RandomSizedCrop(min_max_height=(400, 400), height=512, width=512, p=0.5),

            A.RandomRotate90(p=0.5),

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.Resize(height=512, width=512, p=1),

            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),

            ToTensorV2(p=1.0),                  

        ], p=1.0)



def get_train_transforms1():

    return A.Compose([



            A.Resize(height=512, width=512, p=1),

            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),

            ToTensorV2(p=1.0),                  

        ], p=1.0)



def get_train_transforms2():

    return A.Compose([



            A.Resize(height=512, width=512, p=1),

            A.CenterCrop(256, 256),

            ToTensorV2(p=1.0),                  

        ], p=1.0)



def get_valid_transforms():

    return A.Compose([

            A.Resize(height=512, width=512, p=1.0),

            ToTensorV2(p=1.0),

        ], p=1.0)



DATA_PATH = '../input/melanoma-merged-external-data-512x512-jpeg'

TRAIN_ROOT_PATH = f'{DATA_PATH}/512x512-dataset-melanoma/512x512-dataset-melanoma'



def onehot(size, target):

    vec = torch.zeros(size, dtype=torch.float32)

    vec[target] = 1.

    return vec



class DatasetRetriever(Dataset):



    def __init__(self, image_ids, labels, transforms=None):

        super().__init__()

        self.image_ids = image_ids

        self.labels = labels

        self.transforms = transforms



    def __getitem__(self, idx: int):

        image_id = self.image_ids[idx]

        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        #plt.imshow(image)

        #image = image.astype(np.float32) / 255.0



        label = self.labels[idx]



        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']



        target = onehot(2, label)

        return image, target



    def __len__(self) -> int:

        return self.image_ids.shape[0]



    def get_labels(self):

        return list(self.labels)


df_folds = pd.read_csv(f'{DATA_PATH}/folds.csv', index_col='image_id')

train_dataset = DatasetRetriever(

        image_ids=df_folds[df_folds['fold'] != 1].index.values,

        labels=df_folds[df_folds['fold'] != 1].target.values,

        transforms=get_train_transforms(),

    )



train_dataset1 = DatasetRetriever(

        image_ids=df_folds[df_folds['fold'] != 1].index.values,

        labels=df_folds[df_folds['fold'] != 1].target.values,

        transforms=get_train_transforms1(),

    )

train_dataset2 = DatasetRetriever(

        image_ids=df_folds[df_folds['fold'] != 1].index.values,

        labels=df_folds[df_folds['fold'] != 1].target.values,

        transforms=get_train_transforms2(),

    )
image, label = train_dataset[0]

plt.imshow(image.reshape(512,512,3))
image, label = train_dataset1[0]

plt.imshow(image.reshape(512,512,3))
image, label = train_dataset2[0]

plt.imshow(image.reshape(256,256,3))
import os

os.listdir('../input/ganweight')
img = cv2.imread('../input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma/ISIC_4568001.jpg')



img.shape


json_file = open('../input/ganweight/generator.json', 'r')

generator_json = json_file.read()

json_file.close()

generator = model_from_json(generator_json)

generator.load_weights('../input/ganweight/generator_weights.hdf5')
plt.imshow(img[:,:,1])
originals = img



#data = np.load(".../HAMNOAUG_256.npz")



#labels = 1



#temp = np.empty((0, 128, 128, 3))

for i in range(originals.shape[0]):

    temp_r = skimage.measure.block_reduce(originals[:,:,0], (4,4), np.mean) # (4,4) = factor of reduction

    temp_g = skimage.measure.block_reduce(originals[:,:,1], (4,4), np.mean)

    temp_b = skimage.measure.block_reduce(originals[:,:,2], (4,4), np.mean)

    temp_rgb = np.stack([temp_r, temp_g, temp_b], axis=-1)



    #temp[i] = temp_rgb    

originals = temp_rgb

originals /= 255
nn_sampled_labels = np.concatenate([np.zeros(3), np.ones(3), np.ones(3)+1])

nn_noise = np.random.normal(0, 1, (9, 128))
gen_imgs = 0.5*generator.predict([nn_noise, nn_sampled_labels]) + 0.5
originals.shape
plt.imshow(img)
n_row = 3

n_col = 3

_, axs = plt.subplots(n_row, n_col, figsize=(15, 15))

axs = axs.flatten()



for i in range(9):

    axs[i].imshow(gen_imgs[i])

plt.show()