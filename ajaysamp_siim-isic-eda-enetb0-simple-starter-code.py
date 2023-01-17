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
root_jpeg = '/kaggle/input/siim-isic-melanoma-classification/jpeg/'

train_path_jpeg = root_jpeg + 'train/'

test_path_jpeg = root_jpeg + 'test/'



dcm_root = '/kaggle/input/siim-isic-melanoma-classification/'

dcm_train =  dcm_root +'train/'

dcm_test = dcm_root + 'test/'
# import required libraries

import numpy as np





import PIL

from PIL import Image



# plotly libraries

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import plotly.express as px

import plotly.io as pio

import seaborn as sns

import matplotlib



import pydicom as dicom



from scipy.stats import norm

import random



import cv2



import warnings

warnings.filterwarnings("ignore")
#read train data

train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
train_df.shape[0], train_df.patient_id.nunique()
benign_malig = train_df.groupby('benign_malignant').agg(count=('benign_malignant','count'))

benign_malig.reset_index(inplace=True, drop=False)



fig,ax = plt.subplots(figsize=(8,4))

ax = sns.barplot(x='count',y='benign_malignant',data=benign_malig.sort_values(by='count', ascending=False), 

                 palette = {'benign':'#27AE60','malignant':'#E74C3C'})

ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.title('Benign vs Malignant')

plt.xlabel('')

plt.ylabel('')

plt.tight_layout()

plt.show()
male_female = train_df.groupby('sex').agg(count=('patient_id','nunique'))

male_female.reset_index(inplace=True, drop=False)



fig,ax = plt.subplots(figsize=(8,4))

ax = sns.barplot(x ='count', y = 'sex',data=male_female,palette={'male':'#3498DB','female':'#E74C3C'})

plt.xlabel('', fontsize=20)

plt.ylabel('')

plt.yticks(fontsize=20)

plt.xticks(fontsize=15)

plt.tight_layout()

plt.show()
age_distribution = train_df.groupby('patient_id').agg(sex=('sex','first'),age=('age_approx','first'))

age_distribution.reset_index(inplace=True, drop=False)



fig = plt.figure(figsize = (18,4))

ax = fig.add_subplot(1, 3, 1)

ax = sns.distplot(age_distribution['age'],fit=norm, color='#1ABC9C')

plt.title('Age Distribution - Overall')



ax = fig.add_subplot(1, 3, 2)

ax = sns.distplot(age_distribution[age_distribution.sex == 'male']['age'],fit=norm, color='#3498DB')

plt.title('Age Distribution - Males')



ax = fig.add_subplot(1, 3, 3)

ax = sns.distplot(age_distribution[age_distribution.sex == 'female']['age'],fit=norm, color='#E74C3C')

plt.title('Age Distribution - Females')



plt.tight_layout()

plt.show()
fig,ax = plt.subplots(figsize=(18,4))

ax = sns.boxplot(x ='age', y = 'sex',data=age_distribution,palette={'male':'#3498DB','female':'#E74C3C'})

plt.xlabel('Age', fontsize=20)

plt.ylabel('')

plt.yticks(fontsize=20)

plt.tight_layout()

plt.show()
site = train_df.groupby('anatom_site_general_challenge').agg(count=('anatom_site_general_challenge','count'))

site.reset_index(inplace=True, drop=False)



fig,ax = plt.subplots(figsize=(10,6))

ax = sns.barplot(x='count',y='anatom_site_general_challenge',data=site.sort_values(by='count', ascending=False), color = '#2C3E50')

ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.title('Lesion Count by Site')

plt.xlabel('')

plt.ylabel('')

plt.tight_layout()

plt.show()
diagnosis = train_df.groupby('diagnosis').agg(count=('diagnosis','count'))

diagnosis.reset_index(inplace=True, drop=False)



fig,ax = plt.subplots(figsize=(10,6))

ax = sns.barplot(x='count',y='diagnosis',data=diagnosis.sort_values(by='count', ascending=False), color = '#5DADE2')

ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.title('Diagnosis')

plt.xlabel('')

plt.ylabel('')

plt.tight_layout()

plt.show()
# get the list of image names

images_list = train_df.image_name.values
def plot_image(images,title=None):

    fig = plt.figure(figsize = (20,12))

    for i in range(1,10):

        ax = fig.add_subplot(3, 3, i)

        img = np.random.choice(images)

        image_path = os.path.join(train_path_jpeg,img +'.jpg')

        image = Image.open(image_path)

        ax.imshow(image)

    plt.suptitle(title, fontsize=15)    

    plt.tight_layout()
plot_image(images_list)
benign_list = train_df[train_df.target == 0].image_name.values

malignant_list = train_df[train_df.target == 1].image_name.values
plot_image(benign_list, title='Benign')
plot_image(malignant_list, title='Malignant')
def plot_dcm(images):

    fig = plt.figure(figsize = (20,12))

    for i in range(1,4):

        ax = fig.add_subplot(1, 3, i)

        img = np.random.choice(images)

        image_path = os.path.join(dcm_train,img +'.dcm')

        ds = dicom.dcmread(image_path)

        ax.imshow(ds.pixel_array)   

        plt.tight_layout()
plot_dcm(images_list)
import albumentations as A

from albumentations import (

    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,

    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,

    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,

    IAASharpen, IAAEmboss, Flip, OneOf, Compose,VerticalFlip,BboxParams,Rotate, ChannelShuffle, RandomRain)
def aug_show(image,aug,title=None):

    fig = plt.figure(figsize = (20,12))

    image_path = os.path.join(train_path_jpeg,image +'.jpg')

    image = Image.open(image_path)

    

    ax = fig.add_subplot(1, 2, 1)

    ax.imshow(image)

    

    aug_image = aug(image=image)['image']

    ax = fig.add_subplot(1, 2, 2)

    ax.imshow(image)

    

    plt.suptitle(title, fontsize=15)    

    plt.tight_layout()
BOX_COLOR = (255, 0, 0)

TEXT_COLOR = (255, 255, 255)



def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2, **kwargs):

    #height, width = img.shape[:2]



    x_min, y_min, w, h = bbox

    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    return img



def visualize_titles(img, bbox, title, color=BOX_COLOR, thickness=2, font_thickness = 2, font_scale=0.35, **kwargs):

    #height, width = img.shape[:2]

    x_min, y_min, w, h = bbox

    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    

    ((text_width, text_height), _) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)

    cv2.putText(img, title, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR,

                font_thickness, lineType=cv2.LINE_AA)

    return img





def augment_and_show(aug, image, mask=None, bboxes=[], categories=[], category_id_to_name=[], filename=None, title=None,

                     font_scale_orig=0.35, 

                     font_scale_aug=0.35, show_title=True, **kwargs):



    augmented = aug(image=image, mask=mask, bboxes=bboxes, category_id=categories)



    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)



    for bbox in bboxes:

        visualize_bbox(image, bbox, **kwargs)



    for bbox in augmented['bboxes']:

        visualize_bbox(image_aug, bbox, **kwargs)



    if show_title:

        for bbox,cat_id in zip(bboxes, categories):

            visualize_titles(image, bbox, category_id_to_name[cat_id], font_scale=font_scale_orig, **kwargs)

        for bbox,cat_id in zip(augmented['bboxes'], augmented['category_id']):

            visualize_titles(image_aug, bbox, category_id_to_name[cat_id], font_scale=font_scale_aug, **kwargs)



    

    if mask is None:

        f, ax = plt.subplots(1, 2, figsize=(16, 8))

        

        ax[0].imshow(image)

        ax[0].set_title('Original image')

        

        ax[1].imshow(image_aug)

        ax[1].set_title(title)

    else:

        f, ax = plt.subplots(2, 2, figsize=(16, 16))

        

        if len(mask.shape) != 3:

            mask = label2rgb(mask, bg_label=0)            

            mask_aug = label2rgb(augmented['mask'], bg_label=0)

        else:

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            mask_aug = cv2.cvtColor(augmented['mask'], cv2.COLOR_BGR2RGB)

            

        ax[0, 0].imshow(image)

        ax[0, 0].set_title('Original image')

        

        ax[0, 1].imshow(image_aug)

        ax[0, 1].set_title('Augmented image')

        

        ax[1, 0].imshow(mask, interpolation='nearest')

        ax[1, 0].set_title('Original mask')



        ax[1, 1].imshow(mask_aug, interpolation='nearest')

        ax[1, 1].set_title('Augmented mask')



    f.tight_layout()

    if filename is not None:

        f.savefig(filename)

        

    return augmented['image'], augmented['mask'], augmented['bboxes']
random.seed(42)

image_id = 'ISIC_2245325'

image = cv2.imread(os.path.join(train_path_jpeg,image_id +'.jpg'))



flip = A.Compose([VerticalFlip(p=1)],p=1)



r = augment_and_show(flip, image, title='Vertical Flip')
strong = A.Compose([

    A.ChannelShuffle(p=1),

], p=1)



r = augment_and_show(strong, image, title='Channel Shuffle')
light = A.Compose([

    A.RandomBrightnessContrast(p=1),    

    A.RandomGamma(p=1),    

    A.CLAHE(p=1),    

], p=1)



r = augment_and_show(light, image, title='Adjust Contrast')
# install efficient net

!pip install -q efficientnet
from keras_preprocessing.image import ImageDataGenerator



import efficientnet.tfkeras as efnt



from tensorflow.keras import models

from tensorflow.keras import layers

import tensorflow as tf
# read data

df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')



# get unique ids of patients and shuffle

unique_ids = df['patient_id'].unique()

random.shuffle(unique_ids)



# create a list for training and validation id's  - we will use 20% of the data for validation

train_ids = unique_ids[:int(0.8 * len(unique_ids))]

valid_ids = unique_ids[int(0.8 * len(unique_ids)):]



# check if there are same patient id's both in training and validation sets

print ('Number of Repeating ids:' ,len(set(train_ids).intersection(set(valid_ids))))



# get the train and validation df's

train_df = df[df['patient_id'].isin(train_ids)]

valid_df = df[df['patient_id'].isin(valid_ids)]



print('Train Data Size: ',train_df.shape[0])

print('Validation Data Size: ',valid_df.shape[0])
# add jpg extensions for filename

train_df.loc[:,'image_name'] = train_df['image_name']+'.jpg'

valid_df.loc[:,'image_name'] = valid_df['image_name'] + '.jpg'



# convert target column to string to input into image generator

train_df['target'] = train_df['target'].astype(str)

valid_df['target'] = valid_df['target'].astype(str)


batch_size = 48



width = 512

height = 512

epochs = 10

NUM_TRAIN = train_df.shape[0]

NUM_TEST = valid_df.shape[0]

dropout_rate = 0.2

input_shape = (height, width, 3)
# we will use some in-built augmentations for now

train_datagen=ImageDataGenerator(rescale=1./255, rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,zoom_range=0.2)



train_generator=train_datagen.flow_from_dataframe(dataframe=train_df, directory=train_path_jpeg, 

                                            x_col="image_name", y_col="target", class_mode="binary", 

                                            target_size=(height,width), batch_size=batch_size)



# The validation datatset should not be augmented!

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)



valid_generator=valid_datagen.flow_from_dataframe(dataframe=valid_df, directory=train_path_jpeg, 

                                            x_col="image_name", y_col="target", class_mode="binary", 

                                            target_size=(height,width), batch_size=batch_size)
# load the pretrained model. Set include_top to false since we have only 2 classes

conv_base = efnt.EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
model = models.Sequential()

model.add(conv_base)

model.add(layers.GlobalMaxPooling2D(name="gap"))



if dropout_rate > 0:

    model.add(layers.Dropout(dropout_rate, name="dropout_out"))

    

model.add(layers.Dense(2, activation="softmax", name="fc_out"))
conv_base.trainable = False
model.compile(

    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.1),

    optimizer='adam',

    metrics=["binary_crossentropy"])
#history = model.fit_generator(

 #   train_generator,

  #  steps_per_epoch=20,  # typically the len(train_data)/batch_size. Used a small number for demo.

  #  epochs=3,            # this also has to be tuned. Using a small number just for demo and saving up GPU time

  #  validation_data=valid_generator,

  #  validation_steps=20,

  #  verbose=1)
os.makedirs("./models", exist_ok=True)

model.save('./models/melanoma_base.h5')
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_x = range(len(loss))



plt.plot(epochs_x, loss, 'bo', label='Training loss')

plt.plot(epochs_x, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()