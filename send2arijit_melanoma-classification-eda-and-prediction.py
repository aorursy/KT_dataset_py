# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from os import listdir



import matplotlib.pyplot as plt

%matplotlib inline



#plotly

!pip install chart_studio

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



import seaborn as sns

sns.set(style="whitegrid")



from sklearn.model_selection import train_test_split



#pydicom

import pydicom



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')





# Settings for pretty nice plots

plt.style.use('fivethirtyeight')

plt.show()
import cv2

from kaggle_datasets import KaggleDatasets

IMAGE_PATH='/kaggle/input/siim-isic-melanoma-classification/'

TRAIN_IMG_PATH='/kaggle/input/siim-isic-melanoma-classification/train/'

TRAIN_IMG_JPG_PATH='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

TEST_IMG_JPG_PATH='/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'
# https://www.kaggle.com/schlerp/getting-to-know-dicom-and-the-data

def show_dcm_info(dataset):

    print("Filename.........:", file_name)

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
def plot_pixel_array(dataset, figsize=(5,5)):

    plt.figure(figsize=figsize)

    plt.grid(False)

    plt.imshow(dataset.pixel_array)

    plt.show()

    

i = 1

num_to_plot = 5

for file_name in os.listdir(TRAIN_IMG_PATH):

        file_path = TRAIN_IMG_PATH+file_name

        dcm_data = pydicom.dcmread(file_path)

        show_dcm_info(dcm_data)

        plot_pixel_array(dcm_data)

    

        if i >= num_to_plot:

            break

    

        i += 1
train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
train_df.head()
test_df.head()
print('train data shape: {}'.format(train_df.shape))
train_df.info()
# check for NaN values

train_df.isnull().sum()
train_df['sex'].value_counts()
train_df['sex'].value_counts(normalize=True).iplot(kind='bar', yTitle='percentage', bargap=0.8, title="Sex distribution in the training data")
print("There are {} unique patients among {} records".format(len(train_df['patient_id'].unique()), train_df.shape[0]))
train_df['patient_id'].value_counts().hist(bins=7)
train_df['age_approx'].value_counts(sort=True)
train_df['age_approx'].iplot(kind='hist', yTitle='count', title="Age distribution in the training data")
anatom=train_df['anatom_site_general_challenge'].value_counts()
fig = plt.figure(figsize=(10, 5))

ax = fig.add_axes([0,0,1,1])



ax.bar(x=anatom.index,height=anatom.values)



plt.title("anatom site of the patients", fontsize=18)

plt.show()
train_df['diagnosis'].value_counts()
fig = plt.figure(figsize=(15, 5))

ax = fig.add_axes([0,0,1,1])



ax.bar(x=train_df['diagnosis'].value_counts().index,height=train_df['diagnosis'].value_counts().values)



plt.title("diagnosis", fontsize=18)

plt.show()
train_df['benign_malignant'].value_counts()
fig = plt.figure(figsize=(3, 5))

ax = fig.add_axes([0,0,1,1])



ax.bar(x=train_df['benign_malignant'].value_counts().index,height=train_df['benign_malignant'].value_counts().values)



plt.title("benign and malignant distribution in training data", fontsize=10)

plt.show()
train_df['target'].value_counts()
fig = plt.figure(figsize=(1, 5))

ax = fig.add_axes([0,0,1,1])



ax.bar(x=train_df['target'].value_counts().index,height=train_df['target'].value_counts().values)



plt.title("target distribution", fontsize=10)

plt.show()
train_images=train_df['image_name'].values
# plot the dcm Images

fig=plt.figure(figsize=(15, 10))

columns = 3; rows = 4

plt.title("Digital Imaging and Communications in Medicine (DICOM) images", fontsize=14)

for i in range(1, columns*rows +1):

    ds = pydicom.dcmread(TRAIN_IMG_PATH + train_images[i]+'.dcm')

    fig.add_subplot(rows, columns, i)

    plt.imshow(-ds.pixel_array, cmap=plt.cm.bone)

    fig.add_subplot

fig.tight_layout()

# plot the JPEG Images

fig=plt.figure(figsize=(15, 10))

columns = 3; rows = 4

plt.title("JPEG Images", fontsize=18)



for i in range(1, columns*rows +1):

    img = plt.imread(TRAIN_IMG_JPG_PATH + train_images[i]+'.jpg')

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

    fig.add_subplot

# plot the JPEG Images

fig=plt.figure(figsize=(15, 10))

columns = 3; rows = 4

plt.title("Benign images", fontsize=18)



for i in range(1, columns*rows +1):

    if train_df.iloc[i].benign_malignant=='benign':

        img = plt.imread(TRAIN_IMG_JPG_PATH + train_images[i]+'.jpg')

        fig.add_subplot(rows, columns, i)

        plt.imshow(img)

        fig.add_subplot
# plot the JPEG Images

fig=plt.figure(figsize=(15, 10))

columns = 3; rows = 4

plt.title("Malignant images", fontsize=18)

idx=0

i=1

while idx < train_df.shape[0]:

    if i < (columns*rows + 1):

        if train_df.iloc[idx].benign_malignant=='malignant':

            img = plt.imread(TRAIN_IMG_JPG_PATH + train_images[idx]+'.jpg')

            fig.add_subplot(rows, columns, i)

            plt.imshow(img)

        

            fig.add_subplot

            i = i + 1

    idx=idx+1
test_images=test_df['image_name'].values



# plot the JPEG Images

fig=plt.figure(figsize=(15, 10))

columns = 3; rows = 4

plt.title("Test images", fontsize=18)



for i in range(1, columns*rows +1):

    img = plt.imread(TEST_IMG_JPG_PATH + test_images[i]+'.jpg')

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

    fig.add_subplot
# https://www.kaggle.com/vatsalparsaniya/melanoma-hair-remove

def hair_remove(image):

    # convert image to grayScale

    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    

    # kernel for morphologyEx

    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(17,17))

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(20,20))

    

    # apply MORPH_BLACKHAT to grayScale image

    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    

    # apply thresholding to blackhat

    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

    

    # inpaint with original image and threshold image

    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)

    

    return final_image
# Cross-shaped Kernel

cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
#ISIC_0078712

hair_img1 = plt.imread(TRAIN_IMG_JPG_PATH +'ISIC_0078712'+'.jpg')

plt.imshow(hair_img1)

plt.title("Image with hairs", fontsize=18)
#%%time

## image with hairs removed for 1024,1024 image

image_resize = cv2.resize(hair_img1,(1024,1024))

final_image = hair_remove(image_resize)



plt.imshow(final_image)

plt.title("Hairs removed", fontsize=18)
normal_img1 = plt.imread(TRAIN_IMG_JPG_PATH +'ISIC_0052212'+'.jpg')

plt.imshow(normal_img1)

plt.title("No hair image", fontsize=18)
## image with hairs removed.

## for 1024,1024 image

image_resize = cv2.resize(normal_img1,(1024,1024))

final_image = hair_remove(image_resize)



plt.imshow(final_image)

plt.title("After hair removal process", fontsize=18)
#!pip install tf-explain

!pip install -q efficientnet
import tensorflow as tf

#from tf_explain.core.activations import ExtractActivations

#from tensorflow.keras.applications.xception import decode_predictions

from sklearn.utils import class_weight

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping 

import efficientnet.tfkeras as efn 
x_train, x_val = train_test_split(train_df, test_size=0.2, random_state=42)

x_train['image_name'] = x_train['image_name'].apply(lambda x: x + '.jpg')

x_val['image_name'] = x_val['image_name'].apply(lambda x: x + '.jpg')

test_df['image_name'] = test_df['image_name'].apply(lambda x: x + '.jpg')

x_train['target'] = x_train['target'].apply(lambda x: str(x))

x_val['target'] = x_val['target'].apply(lambda x: str(x))
# detect TPU

DEVICE='TPU'

if DEVICE == "TPU":

    print("connecting to TPU...")

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        print('Running on TPU ', tpu.master())

    except ValueError:

        print("Could not connect to TPU")

        tpu = None



    if tpu:

        try:

            print("initializing  TPU ...")

            tf.config.experimental_connect_to_cluster(tpu)

            tf.tpu.experimental.initialize_tpu_system(tpu)

            strategy = tf.distribute.experimental.TPUStrategy(tpu)

            print("TPU initialized")

        except _:

            print("failed to initialize TPU")

    else:

        DEVICE = "GPU"



if DEVICE != "TPU":

    print("Using default strategy for CPU and single GPU")

    strategy = tf.distribute.get_strategy()



if DEVICE == "GPU":

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    



AUTO     = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync

print(f'REPLICAS: {REPLICAS}')
"""

IMG_HEIGHT = 256

IMG_WIDTH = 256

N_CHANNELS = 3

epochs = 16

BATCH_SIZE = 16 * REPLICAS

IMAGE_SIZE = [IMG_HEIGHT, IMG_WIDTH]

IMAGE_RESIZE = [IMG_HEIGHT, IMG_WIDTH]

input_shape = (IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)

BALANCE_DATA = True

aug_data = True

NETWORK_MODEL = 'EfficientNetB0'

"""
"""

#train_df = pd.read_csv(base_dir + 'train.csv')

y_train = train_df['target']



class_weights = class_weight.compute_class_weight('balanced',

                                                 classes=np.unique(y_train),

                                                 y=y_train)







class_weights = {0: class_weights[1],1: class_weights[0]}

if not BALANCE_DATA:

    class_weights = {0: 1,1: 1}

print(class_weights)

"""