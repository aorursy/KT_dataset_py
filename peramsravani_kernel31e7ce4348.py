import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob # For pathname matching
from skimage.transform import resize
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model 
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten,concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import cv2

from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
#from scipy.misc import imresize

from time import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from matplotlib.pyplot import rc
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 12}
rc('font', **font)  # pass in the font dict as kwargs
#K.set_image_dim_ordering('th')
import os
from os.path import basename
print(os.listdir("../input"))
print(os.listdir("../"))
input_folder = '../input/aerial-images1'

train= glob('/'.join([input_folder,'train/*.jpg']))
train_masks= glob('/'.join([input_folder,'train_masks/*.gif']))
test= glob('/'.join([input_folder,'test/*.jpg']))
print('Number of training images: ', len(train), 'Number of corresponding masks: ', len(train_masks), 'Number of test images: ', len(test))
tt_ratio = 0.8
img_rows, img_cols = 1024,1024
batch_size = 8
def dice_coef(y_true, y_pred, smooth=0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection=K.sum(y_true_f * y_pred_f)
    return(2. * intersection + smooth) / ((K.sum(y_true_f) + K.sum(y_pred_f)) + smooth)
#split the training set into train and validation samples
train_images, validation_images = train_test_split(train, train_size=tt_ratio, test_size=1-tt_ratio)
print('Size of the training sample=', len(train_images), 'and size of the validation sample=', len(validation_images), ' images')
train=pd.read_csv("https://drive.google.com/drive/folders/1Nrh06A5SQxEnLBwEBYskSZ72kCBmHwWk?usp=sharing")
train.head()
test=pd.read_csv("https://drive.google.com/drive/folders/1xy5Y3r7ZDknBibWeArNF7AinX7tI_rnC?usp=sharing")
test.head()
validation=pd.read_csv("https://drive.google.com/drive/folders/13-EYXpo_tkDhAg4wV6LRCidgD9wTtsIt?usp=sharing")
validation.head()
def grey2rgb(img):
    new_img = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img.append(list(img[i][j])*3)
    new_img = np.array(new_img).reshape(img.shape[0], img.shape[1], 3)
    return new_img

#generator that will be used to read data from the directory
def data_generator(data_dir, masks, images, dims, batch_size=batch_size):
    while True:
        ix=np.random.choice(np.arange(len(images)), batch_size)
        imgs = []
        labels = []
        for i in ix:
            # images
            original_img = cv2.imread(images[i])
            resized_img = imresize(original_img, dims + [3]) 
            array_img = resized_img/255
            array_img = array_img.swapaxes(0, 2)
            imgs.append(array_img)
            #imgs is a numpy array with dim: (batch size X 128 X 128 3)
            #print('shape of imgs ', array_img.shape)
            # masks
            try:
                mask_filename = basename(images[i])
                file_name = os.path.splitext(mask_filename)[0]
                correct_mask = '/'.join([input_folder,'train_masks',file_name+'_mask.gif'])
                original_mask = Image.open(correct_mask).convert('L')
                data = np.asarray(original_mask, dtype="int32")
                resized_mask = imresize(original_mask, dims+[3])
                array_mask = resized_mask / 255
                labels.append(array_mask)
            except Exception as e:
                labels=None
                imgs = np.array(imgs)
                labels = np.array(labels)
        try:
            relabel = labels.reshape(-1, dims[0], dims[1], 1)
            relabel = relabel.swapaxes(1, 3)
        except Exception as e:
            relabel=labels
        yield imgs, relabel