# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



import os

import sys

import zipfile



import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image



# Any results you write to the current directory are saved as output.
class_map = pd.read_csv("../input/bengaliai-cv19/class_map.csv")

sample_submission = pd.read_csv("../input/bengaliai-cv19/sample_submission.csv")

test = pd.read_csv("../input/bengaliai-cv19/test.csv")

train = pd.read_csv("../input/bengaliai-cv19/train.csv")
os.makedirs('train_images128')
!ls
os.listdir('../input/bengaliai/256_train/')
folder = '../input/bengaliai/256_train/256'

width  = 128

height  = 128



def load_images_from_folder(folder):

    images = []

    for filename in os.listdir(folder):

        img = cv2.imread(os.path.join(folder,filename))

        dim = (width, height)

        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        cv2.imwrite("/kaggle/working/train_images128/" + filename,resized)

     
%%time

load_images_from_folder(folder)
#taken from : https://www.kaggle.com/xhlulu/recursion-2019-load-resize-and-save-images



def zip_and_remove(path):

    ziph = zipfile.ZipFile(f'{path}.zip', 'w', zipfile.ZIP_DEFLATED)

    

    for root, dirs, files in os.walk(path):

        for file in tqdm(files):

            file_path = os.path.join(root, file)

            ziph.write(file_path)

            os.remove(file_path)

    

    ziph.close()
zip_and_remove('train_images128')