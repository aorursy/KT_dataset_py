import numpy as np

import pandas as pd 

import pydicom

import matplotlib.pyplot as plt

import os

from pathlib import Path

import cv2
#Checking for errorneous files

base_pth = "../input/siim-acr-pneumothorax-segmentation/sample images/"

for each in (os.listdir("../input/siim-acr-pneumothorax-segmentation/sample images")):

    try:

        ds = pydicom.dcmread(base_pth + each)

    except:

        print("Error File: "+each)



#Reading and displaying a sample image



print("\n Sample File looks like: \n")

ds = pydicom.dcmread("../input/siim-acr-pneumothorax-segmentation/sample images/1.2.276.0.7230010.3.1.4.8323329.1000.1517875165.878027.dcm")

plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 

print(ds.pixel_array)
#Prepping X_train

#Note - 256 x 256 - 2.6 GB, 512 x 512 - 4.5 GB

    

X_train = []



for filename in Path('../input/siim-train-test/siim/dicom-images-train/').glob('**/*.dcm'):

    ds = pydicom.dcmread(str(filename))

    b0 = cv2.resize(ds.pixel_array,(512,512))

    b = np.reshape(b0,-1)

    X_train.append(b)
#Prepping Y_train

import re



df_train = pd.read_csv('../input/siim-train-test/siim/train-rle.csv')

y_train = []



for filename in Path('../input/siim-train-test/siim/dicom-images-train/').glob('**/*.dcm'):

    tmp = re.search('/[0-9.]*[.]*[0-9]*.dcm',str(filename))

    idz = str(tmp.group()[1:-4])

    

    for idx,each in enumerate(df_train['ImageId']):

        if each == idz:

            if (df_train.iloc[idx,1] == ' -1'):

                y_train.append(0)

            else:

                y_train.append(1)

        else:

            continue