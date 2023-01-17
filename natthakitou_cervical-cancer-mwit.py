import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

      
#Any results you write to the current directory are saved as output
#import all the used function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

#pip install opencv-python
#!pip install keras
#!pip install --upgrade "tensorflow==1.7.*"
#!pip install tensorflow


#unzipping data before using
#local_zip = '/kaggle/input/intel-mobileodt-cervical-cancer-screening/'
#zip_ref = zipfile.ZipFile(local_zip, 'r')
#zip_ref.extractall('/kaggle/input/intel-mobileodt-cervical-cancer-screening/')
#zip_ref.close()
print(os.listdir("../input/intel-mobileodt-cervical-cancer-screening/train/train"))
#/kaggle/input/intel-mobileodt-cervical-cancer-screening/additional/  as dataset
#os.listdir(rock_dir)
from subprocess import check_output
print(check_output(["ls", "../input/intel-mobileodt-cervical-cancer-screening/"]).decode("utf8"))
from glob import glob
#separate data
TRAIN_DATA = "../input/intel-mobileodt-cervical-cancer-screening/train/train"
type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1"))+1:-4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2"))+1:-4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3"))+1:-4] for s in type_3_files])

print(len(type_1_files), len(type_2_files), len(type_3_files))
print("Type 1", type_1_ids[:10])
print("Type 2", type_2_ids[:10])
print("Type 3", type_3_ids[:10])
#test data set

TEST_DATA = "../input/intel-mobileodt-cervical-cancer-screening/test/test"
test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = np.array([s[len(TEST_DATA)+1:-4] for s in test_files])
print(len(test_ids))
print(test_ids[:10])

#additional data set
ADDITIONAL_DATA = "../input/intel-mobileodt-cervical-cancer-screening/"
additional_type_1_files = glob(os.path.join(ADDITIONAL_DATA, "additional_Type_1_v2/Type_1", "*.jpg"))
additional_type_1_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "additional_Type_1_v2/Type_1"))+1:-4] for s in additional_type_1_files])
additional_type_2_files = glob(os.path.join(ADDITIONAL_DATA, "additional_Type_2_v2/Type_2", "*.jpg"))
additional_type_2_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "additional_Type_2_v2/Type_2"))+1:-4] for s in additional_type_2_files])
additional_type_3_files = glob(os.path.join(ADDITIONAL_DATA, "additional_Type_3_v2/Type_3", "*.jpg"))
additional_type_3_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "additional_Type_3_v2/Type_3"))+1:-4] for s in additional_type_3_files])


print(len(additional_type_1_files), len(additional_type_2_files), len(additional_type_3_files))
print("Type 1", additional_type_1_ids[:10])
print("Type 2", additional_type_2_ids[:10])
print("Type 3", additional_type_3_ids[:10])
#try cv2
import cv2

img=cv2.imread(test_files[1])
cv2.imshow("img",img)
cv2.waitKey(0)

#crop the cervix
#using cv2 to detect circle and draw rectangle










