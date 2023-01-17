# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")

test_df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
train_df.head()
test_df.head()
print("Class count for training dataset \n",pd.Index(train_df['benign_malignant']).value_counts())
train_df.isna().sum()
train_df.patient_id.nunique()
train_df['target'].value_counts()
train_df['anatom_site_general_challenge'].value_counts()
train_df['diagnosis'].value_counts()
import random

IMAGE_PATH = "../input/siim-isic-melanoma-classification/"

images = train_df['image_name'].values

random_images = [np.random.choice(images+'.jpg') for i in range(6)]

img_dir = IMAGE_PATH+'/jpeg/train'

plt.figure(figsize=(10,8))

for i in range(6):

    plt.subplot(2, 3, i + 1)

    img = plt.imread(os.path.join(img_dir, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

plt.tight_layout()  
import pydicom

def show_dcm_info(dataset):

    print("Filename.........:", file_path)

    print("Storage type.....:", dataset.SOPClassUID)

    print()



    pat_name = dataset.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    print("Patient's name......:", display_name)

    print("Patient id..........:", dataset.PatientID)

    print("Patient's Age.......:", dataset.PatientAge)

    print("Patient's Sex.......:", dataset.PatientSex)

    print("Modality............:", dataset.Modality)

    print("Body Part Examined..:", dataset.BodyPartExamined)

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing....:", dataset.PixelSpacing)
def plot_pixel_array(dataset, figsize=(10,10)):

    plt.figure(figsize=figsize)

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.show()
i = 1

num_to_plot = 2

for file_name in os.listdir('../input/siim-isic-melanoma-classification/train/'):

    file_path = os.path.join('../input/siim-isic-melanoma-classification/train/', file_name)

    dataset = pydicom.dcmread(file_path)

    show_dcm_info(dataset)

    plot_pixel_array(dataset)

    

    if i >= num_to_plot:

        break

    

    i += 1