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

        #print(os.path.join(dirname, filename)) # otherwise too long

        continue



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

%matplotlib inline
data_path = Path('/kaggle/input/siim-isic-melanoma-classification/')

train_path = data_path / 'train'

test_path = data_path / 'test'

print("training_path", train_path)

print("test_path", test_path)
!ls /kaggle/input/siim-isic-melanoma-classification/
#!ls /kaggle/input/siim-isic-melanoma-classification/jpeg/train
#!ls /kaggle/input/siim-isic-melanoma-classification/jpeg/test
data_path = Path('/kaggle/input/siim-isic-melanoma-classification/')

im_train_path = data_path / 'jpeg' / 'train'

im_test_path = data_path / 'jpeg' / 'test'

print("train_path: ", im_train_path)

print("test_path:  ", im_test_path)
import tensorflow as tf

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



from keras import models, regularizers, layers, optimizers, losses, metrics

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils, to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image
def image_show(im_num,im_folder,im_size):

    """

    MO: Show melanoma images.

    """

    im_ind = 'ISIC'

    im_name = '{}_{}'.format(im_ind,im_num)

    if im_folder=='train':

        im_dir = im_train_path

    elif im_folder=='test':

        im_dir = im_test_path

    im_path = str(im_dir)+'/'+str(im_name)+'.jpg'

    im_path

    

    #from tf.keras.preprocessing.image.load_img

    img = image.load_img(im_path, target_size=(im_size, im_size)) #target_size=(224, 224)

    imgplot = plt.imshow(img)

    print(im_ind,"Image Number:", im_num)

    plt.show()
plt.figure(figsize = (10,10))

image_show(5225336,'train',224)
plt.figure(figsize = (10,10))

image_show(5224960,'test',224)
!ls /kaggle/input/siim-isic-melanoma-classification/test/ISIC_5224960.dcm
#import dicom



import pydicom

from pydicom.data import get_testdata_files



print(__doc__)



PathDicom = '/kaggle/input/siim-isic-melanoma-classification/'

lstFilesDCM = []  # create an empty list

for dirName, subdirList, fileList in os.walk(PathDicom):

    for filename in fileList:

        if ".dcm" in filename.lower():  # check whether the file's DICOM

            lstFilesDCM.append(os.path.join(dirName,filename))
print(lstFilesDCM[0])
RefDs = pydicom.dcmread(lstFilesDCM[0])

RefDs
# Get ref file

RefDs = pydicom.dcmread(lstFilesDCM[0])



# Load dimensions based on the number of rows, columns, and slices (along the Z axis)

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

print(ConstPixelDims)
pat_name = RefDs.PatientName

display_name = pat_name.family_name + ", " + pat_name.given_name

print("Patient's name...:", display_name)

print("Patient id.......:", RefDs.PatientID)

print("Modality.........:", RefDs.Modality)

print("Study Date.......:", RefDs.StudyDate)
if 'PixelData' in RefDs:

    rows = int(RefDs.Rows)

    cols = int(RefDs.Columns)

    print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(rows=rows, cols=cols, size=len(RefDs.PixelData)))

    if 'PixelSpacing' in RefDs:

        print("Pixel spacing....:", RefDs.PixelSpacing) 



        

# use .get() if not sure the item exists, and want a default value if missing

print("Slice location...:", RefDs.get('SliceLocation', "(missing)"))



# plot the image using matplotlib

plt.figure(figsize = (10,10))

plt.imshow(RefDs.pixel_array, cmap=plt.cm.bone)

plt.show()
train = pd.read_csv(data_path / 'train.csv')

test  = pd.read_csv(data_path / 'test.csv')

sub   = pd.read_csv(data_path / 'sample_submission.csv')



train.shape, test.shape, sub.shape
train.isna().sum()
train['sex'] = train['sex'].fillna('na')

train['age_approx'] = train['age_approx'].fillna(0)

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')
train.isna().sum()
train.head(10)
test.isna().sum()
test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')
test.isna().sum()
test.head(10)
train['sex'].value_counts().plot(kind='bar')
test['sex'].value_counts().plot(kind='bar')
train['sex'].isna().sum()
train['age_approx'].value_counts().plot(kind='bar')
test['age_approx'].value_counts().plot(kind='bar')
train['diagnosis'].value_counts().plot(kind='bar')
train['diagnosis'].value_counts()
print('Diagnosis                             Percent\n-----------------------------------------------')

print((train['diagnosis'].value_counts() / train['diagnosis'].value_counts().sum() ) *100)
fig, axs = plt.subplots(4,2, figsize=(13,20))



# left train, right test



train['sex'].value_counts().plot(kind='bar', legend=True, ax=axs[0,0])

test['sex'].value_counts().plot(kind='bar', legend=True, ax=axs[0,1])



train['age_approx'].value_counts().plot(kind='bar', legend=True, ax=axs[1,0])

test['age_approx'].value_counts().plot(kind='bar', legend=True, ax=axs[1,1])



train['age_approx'].hist(bins=90, ax=axs[2,0])

test['age_approx'].hist(bins=90, ax=axs[2,1])

axs[2,0].set_xlabel('Age')

axs[2,1].set_xlabel('Age')



train['anatom_site_general_challenge'].value_counts().plot(kind='bar', legend=True, ax=axs[3,0])

test['anatom_site_general_challenge'].value_counts().plot(kind='bar', legend=True, ax=axs[3,1])





plt.savefig('data_sex_age_anatom.png',dpi=100)



plt.show()