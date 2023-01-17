import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copy
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, add, Dropout, Flatten, Dense, Reshape, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from zipfile import ZipFile
import sys
from tensorflow.keras.models import Sequential, load_model
from  tensorflow import keras
from sklearn.metrics import roc_curve, roc_auc_score
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
allData = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv', sep = ',')
df = pd.DataFrame(allData)
malignant = df.groupby(df.target).get_group(1) 
benign = df.groupby(df.target).get_group(0)

malignant_file_names = malignant.image_name.values
benign_file_names = benign.image_name.values
print(len(malignant_file_names))

#make the required directories
dirs = ['Data', 'Data/train', 'Data/test', 'Data/train/benign', 'Data/train/malignant', 'Data/test/benign', 'Data/test/malignant']

for dir in dirs:
    os.mkdir(dir)
#Copy benigns
source = '../input/resize-jpg-siimisic-melanoma-classification/300x300/train'
traindest = 'Data/train/benign' 
testdest = 'Data/test/benign' 
trainSize = 32000
testSize = 100
benign_file_names_train = benign_file_names[:trainSize]
benign_file_names_test = benign_file_names[trainSize:trainSize+testSize]

for i,file in enumerate(benign_file_names_train):
  os.system('cp -r %s %s'%(source+'/'+file+'.jpg', traindest+'/'+file+'.jpg'))
  sys.stdout.write('\r %d%%: Copying %s to %s'%((i/len(benign_file_names_train))*100, file, traindest))
  sys.stdout.flush()
print("\rComplete")

for i,file in enumerate(benign_file_names_test):
  os.system('cp -r %s %s'%(source+'/'+file+'.jpg', testdest+'/'+file+'.jpg'))
  sys.stdout.write('\r %d%%: Copying %s to %s'%((i/len(benign_file_names_test))*100, file, testdest))
  sys.stdout.flush()
print("\rComplete")
#Copy malignants
source = '../input/resize-jpg-siimisic-melanoma-classification/300x300/train'
traindest = 'Data/train/malignant' 
testdest = 'Data/test/malignant' 
testSize = 50
malignant_file_names_train = malignant_file_names[testSize:]
malignant_file_names_test = malignant_file_names[:testSize]

for i,file in enumerate(malignant_file_names_train):
  os.system('cp -r %s %s'%(source+'/'+file+'.jpg', traindest+'/'+file+'.jpg'))
  sys.stdout.write('\r %d%%: Copying %s to %s'%((i/len(malignant_file_names_train))*100, file, traindest))
  sys.stdout.flush()
print("\rComplete")

for i,file in enumerate(malignant_file_names_test):
  os.system('cp -r %s %s'%(source+'/'+file+'.jpg', testdest+'/'+file+'.jpg'))
  sys.stdout.write('\r %d%%: Copying %s to %s'%((i/len(malignant_file_names_test))*100, file, testdest))
  sys.stdout.flush()
print("\rComplete")
import shutil
shutil.make_archive('Data', 'zip', 'Data')
shutil.rmtree('Data')
