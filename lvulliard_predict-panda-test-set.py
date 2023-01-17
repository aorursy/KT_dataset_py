import os

import shutil



# There are two ways to load the data from the PANDA dataset:

# Option 1: Load images using openslide

import openslide

# Option 2: Load images using skimage (requires that tifffile is installed)

import skimage.io



# General packages

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import PIL

from IPython.display import Image, display

from collections import Counter



import cv2

import skimage.io

from tqdm.notebook import tqdm



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer



from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

tf.test.is_gpu_available()
# Load images: test set if actual submission, train set otherwise

dataTestDir = '/kaggle/input/prostate-cancer-grade-assessment/test_images'



if os.path.exists(dataTestDir):

    # Test set is available

    dataDir = dataTestDir

    labels = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/test.csv').set_index('image_id')

    cropDir = '/kaggle/working/cropped_test_images/'

    # Create this folder

    if not os.path.exists(cropDir):

        os.mkdir(cropDir)

else:

    dataDir = '/kaggle/input/prostate-cancer-grade-assessment/train_images'

    labels = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv').set_index('image_id')

    cropDir = '/kaggle/input/crop-images/cropped_train_images/'
def tile(img):

    result = []

    shape = img.shape

    pad0,pad1 = (cropPx - shape[0]%cropPx)%cropPx, (cropPx - shape[1]%cropPx)%cropPx

    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],

                constant_values=255)

    

    img = img.reshape(img.shape[0]//cropPx,cropPx,img.shape[1]//cropPx,cropPx,3)

    img = img.transpose(0,2,1,3,4).reshape(-1,cropPx,cropPx,3)

    

    if len(img) < cropN:

        img = np.pad(img,[[0,cropN-len(img)],[0,0],[0,0],[0,0]],constant_values=255)

    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:cropN]

    img = img[idxs]

    for i in range(len(img)):

        result.append({'img':img[i], 'idx':i})

    return result
if os.path.exists(dataTestDir):

    # Test set is available



    # Parameters for cropping images

    cropPx= 56

    cropN = 16

    assert np.sqrt(cropN) == round(np.sqrt(cropN))



    nbCol = int(np.sqrt(cropN))

    names = [x.split('.')[0] for x in os.listdir(dataDir)]

    for name in tqdm(names):

        img = skimage.io.MultiImage(os.path.join(dataDir+'/',name+'.tiff'))[-1]

        tiles = tile(img)

        stackImg = np.vstack([np.hstack([tiles[nbCol*col + row]['img'] for row in range(nbCol)])

                   for col in range(nbCol)])

        cv2.imwrite(cropDir+name+'.png', stackImg)
!pip install -qq /kaggle/input/sequencedata/efficientnet-1.1.0-py3-none-any.whl
import efficientnet.tfkeras

from tensorflow.keras.models import load_model
inputShape = (224, 224, 3)



modelFile = "/kaggle/input/fine-tune-efficient-b1-2nd-step/B1_100_r100.model"

myModel = load_model(modelFile)

myModel.summary()
testDF = pd.DataFrame(list(zip(labels.index + ".png")), columns =['x_col']) 

# Uncomment the following to fasten local test (otherwise prediction takes about one minute)

# if not os.path.exists(dataTestDir):

#     testDF = testDF[:2050]



nbSteps = testDF.shape[0]



testDatagen = ImageDataGenerator() 

testGenerator = testDatagen.flow_from_dataframe(testDF, x_col="x_col", y_col=None, directory=cropDir, # this is the target directory 

                                                batch_size=256, class_mode=None, shuffle=False,

                                                target_size=(inputShape[0], inputShape[1]), 

                                                color_mode='rgb')
preds = myModel.predict_generator(testGenerator,steps=nbSteps/256)
outputDF = pd.DataFrame([np.argmax(preds[i,:]) for i in range(nbSteps)], 

                        index = labels.index[:nbSteps], columns = ["isup_grade"])
outputDF.to_csv("../working/submission.csv")