# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
def get_class_labels(directory):

    directory_tokens = directory.split("/")

    img_foldername = directory_tokens[-1]

    

    folder_tokens = img_foldername.split("_")

    

    plant_type = folder_tokens[0]

    disease_type = "_".join(folder_tokens[1:])

        

    return plant_type.lower(), disease_type.lower()

    
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        img = cv2.imread(os.path.join(dirname, filename))

        break

plt.imshow(im)

plt.show()
get_class_labels("/input/plantdisease/PlantVillage/PlantVillage/Tomato_Spider_mites_Two_spotted_spider_mite")
##

#  Sample models to use for 

#  classification: 

#  Resnet50 trained on Imagenet

#  MobileNet, also trained on ImageNet

#  Requires downsampling images to 224 x 224

#  -- https://keras.io/applications/#nasnet

##