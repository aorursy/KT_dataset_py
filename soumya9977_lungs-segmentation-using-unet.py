import os
train_dir='/kaggle/input/depthwise-conv-adv/data_aug/train/content/data/content/FINAL_AUG_DATA/Train/'
test_dir='/kaggle/input/depthwise-conv-adv/data_aug/val/content/data/content/FINAL_AUG_DATA/Val/'
os.listdir(train_dir)
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
# %run -i '../input/gdown-package/Gdown.txt'
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
!pip install albumentations > /dev/null
# !pip install -U efficientnet==0.0.4
!pip install -U segmentation-models

import numpy as np
import pandas as pd
import gc
import keras

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split,StratifiedKFold

from skimage.transform import resize
import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import  ModelCheckpoint
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from tqdm import tqdm_notebook
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.losses import binary_crossentropy
import keras.callbacks as callbacks
from keras.callbacks import Callback
from keras.applications.xception import Xception
from keras.layers import multiply


from keras import optimizers
from keras.legacy import interfaces
from keras.utils.generic_utils import get_custom_objects

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.regularizers import l2
from keras.layers.core import Dense, Lambda
from keras.layers.merge import concatenate, add
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras.optimizers import SGD,Adam
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

import glob
import shutil
import os
import random
from PIL import Image
import cv2

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
# tf.set_random_seed(seed)
    
%matplotlib inline
def dice_loss(y_true,y_pred):   
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coeff = (intersection * 2 + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice_coeff
def bce_dice_loss(y_true,y_pred):
    return binary_crossentropy(y_true,y_pred) + dice_loss(y_true,y_pred)
def get_iou_vector(A,B):
    batch_size = A.shape[0]
    metric = 0.0
    for i in range(batch_size):
        t,p = A[i],B[i]
        intersection = np.sum(t * p)
        true = np.sum(t)
        pred = np.sum(p)
        
        if(true == 0):
            metric += (pred == 0)
            
        union = true + pred - intersection
        iou = intersection / union
        iou = np.floor(max(0,(iou - 0.45) * 20)) / 10
        metric += iou
    return metric / batch_size
def iou_metric(label,pred):
    return tf.compat.v1.py_func(get_iou_vector,[label,pred > 0.5],tf.float64)
dependencies = {
    'iou_metric': iou_metric,
    'bce_dice_loss':bce_dice_loss
}

model = load_model('../input/lung-segmentation-unet/best_model.h5',custom_objects=dependencies)
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('../input/covid19-radiography-database/COVID-19 Radiography Database/NORMAL/NORMAL (670).png')
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img1)

# img.shape
# plt.imshow(img,cmap='gray')
img = np.reshape(img,(1,1024,1024,3))
prediction=model.predict(img)
type(prediction)
pred_img = np.reshape(prediction,(1024, 1024)) 

fig , (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(pred_img)
ax2.imshow(img1)
ax1.legend()
ax2.legend()
ret,thresh = cv2.threshold(pred_img,0.21205534,1.0,cv2.THRESH_BINARY)
plt.imshow(thresh)
print(np.unique(thresh))
arr = thresh.astype('int8') 
show_masked_img = cv2.bitwise_and(img1,img1,mask = arr)
plt.imshow(show_masked_img)
import gc
gc.collect()
DIR_covid = '../input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19'
DIR_normal = '../input/covid19-radiography-database/COVID-19 Radiography Database/NORMAL'
DIR_pneumonia = '../input/covid19-radiography-database/COVID-19 Radiography Database/Viral Pneumonia'

file_covid  = os.listdir(DIR_covid)
file_normal  = os.listdir(DIR_normal)
file_pneumonia  = os.listdir(DIR_pneumonia)

print(len(file_covid))
print(len(file_pneumonia))
print(len(file_normal))

list_covid = []
list_normal = []
list_pneumonia = []

for i in file_covid:
    lists1 = os.path.join(DIR_covid,i)
    list_covid.append(lists1)


for j in file_normal:
    lists2 = os.path.join(DIR_normal,j)
    list_normal.append(lists2)
    

for k in file_pneumonia:
    lists3 = os.path.join(DIR_pneumonia,k)
    list_pneumonia.append(lists3)
    
    
    

main_list = list_covid+list_normal+list_pneumonia
main_list[0]
from tqdm import tqdm
IMG_SIZE = 224
img = []

for i in tqdm(main_list):
    pic = cv2.imread('../input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19/COVID-19(219).png')
    pic1 = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
    
#   plt.imshow(pic1)
    
    pic = np.reshape(pic,(1,1024,1024,3))
    pred = model.predict(pic)
    predicted_img = np.reshape(pred,(1024, 1024)) 
    
#   plt.imshow(predicted_img)
    
    ret1,thresh1 = cv2.threshold(predicted_img,predicted_img.mean(),1.0,cv2.THRESH_BINARY)
    
#   plt.imshow(thresh1)
    
    arr1 = thresh1.astype('int8') 
    segmented_img = cv2.bitwise_and(pic1,pic1,mask = arr1)
    
#   plt.imshow(segmented_img)

    img.append(segmented_img)


    


gc.collect()
# pic = cv2.imread('../input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19/COVID-19(219).png')
# pic1 = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
# # plt.imshow(pic1)
# pic = np.reshape(pic,(1,1024,1024,3))
# pred = model.predict(pic)
# predicted_img = np.reshape(pred,(1024, 1024)) 
# # plt.imshow(predicted_img)
# ret1,thresh1 = cv2.threshold(predicted_img,predicted_img.mean(),1.0,cv2.THRESH_BINARY)
# # plt.imshow(thresh1)
# arr1 = thresh1.astype('int8') 
# segmented_img = cv2.bitwise_and(pic1,pic1,mask = arr1)
# plt.imshow(segmented_img)


covid_label = []
normal_label = []
pneumonia_label = []

for i in range(219):
    zero = 0
    covid_label.append(zero)
    
for i in range(1341):
    one = 1
    normal_label.append(one)
    
for i in range(1345):
    two = 2
    pneumonia_label.append(two)
    
main_label = covid_label + normal_label + pneumonia_label
train_imgs = np.asarray(img)
print(train_imgs.shape)
train_labels =  np.asarray(main_label)
BATCH_SIZE = 64
SEED = 42
EPOCHS = 100
x, y, z = 224, 224, 3
inputShape = (x, y, z)
NUM_CLASSES = 1
!pip install gdown
!/opt/conda/bin/python3.7 -m pip install --upgrade pip
!pip install gdown
import gdown

train_url = 'https://drive.google.com/uc?id=1-4WfgSQLMIMxl-vrqDjZTMiwb1w3qiJq'
val_url = 'https://drive.google.com/uc?id=1w9CuqPi3DbvbCN9DFGwLPeyYsXJvbjzk'
#the problem is that the whole file train+val is too big for kaggle!
#as kaggle gives 4.9gb of stage and the dataset is 2.8 gb
#since we are using zipped file first.
#https://drive.google.com/open?id=1-4WfgSQLMIMxl-vrqDjZTMiwb1w3qiJq --training
#https://drive.google.com/open?id=1w9CuqPi3DbvbCN9DFGwLPeyYsXJvbjzk --validation
output_train = 'dataset_train.zip'
output_val = 'dataset_val.zip'
gdown.download(train_url, output_train, quiet=False)
os.remove(file_name)

import os
def create_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
    print("Created Directory : ", dir)
    return
create_dir("/kaggle/working/data_aug/train")
!unzip -q /kaggle/working/dataset_train.zip -d /kaggle/working/data_aug/train

os.listdir("/kaggle/working/")

file_name="/kaggle/working/dataset_train.zip"
os.remove(file_name)

os.listdir("/kaggle/working/")

gdown.download(val_url, output_val, quiet=False)

os.listdir("/kaggle/working/")


create_dir("/kaggle/working/data_aug/val")
!unzip -q /kaggle/working/dataset_val.zip -d /kaggle/working/data_aug/val


os.listdir("/kaggle/working/data_aug")
file_name="/kaggle/working/dataset_val.zip"
os.remove(file_name)


import os
"""
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""



























