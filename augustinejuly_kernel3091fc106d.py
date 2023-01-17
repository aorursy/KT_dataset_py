import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

import seaborn as sns

import gc

import glob

import os

import cv2

import pydicom



import warnings

warnings.simplefilter(action = 'ignore')
detailed_df = pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')

train_df = pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
## shape of detailed_df

detailed_df.shape
## shape of train_df

train_df.shape
detailed_df.head()
train_df.head()
df = pd.merge(left = detailed_df, right = train_df, how = 'left', on = 'patientId')

df = df.drop_duplicates()

df.info()
df.info()
df.isnull().sum()
pd.pivot_table(df,index=["Target"], values=['patientId'], aggfunc='count')



# alternative approach

# train_df['Target'].value_counts()
pd.pivot_table(df,index=["class"], values=['patientId'], aggfunc='count')
df["class"].value_counts().plot(kind='pie',autopct='%1.0f%%', shadow=True, subplots=False)
pd.pivot_table(df,index=["Target"], aggfunc='count')
df['patientId'].value_counts().value_counts()
df[df['Target'] == 0]['patientId'].value_counts().value_counts()
sns.countplot(x = 'class', hue = 'Target', data = df)
df.fillna(0.0)
df.corr()
sns.jointplot(x = 'width', y = 'height', data = df, kind="reg")
df_Not_Normal = df[df['class']=='No Lung Opacity / Not Normal'].sample(n=390)

df_Normal = df[df['class']=='Normal'].sample(n=290)

df_Lunge_Opacity = df[df['class']=='Lung Opacity'].sample(n=320)

frames = [df_Not_Normal, df_Normal, df_Lunge_Opacity]



dicom_df = pd.concat(frames)



dicom_df.shape
def process_dicom_data(data_df):

    for n, pid in enumerate(data_df['patientId'].unique()):        

        dcm_file = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % pid

        dcm_data = pydicom.read_file(dcm_file)        

        idx = (data_df['patientId']==dcm_data.PatientID)

        data_df.loc[idx,'Modality'] = dcm_data.Modality

        data_df.loc[idx,'PatientAge'] = pd.to_numeric(dcm_data.PatientAge)

        data_df.loc[idx,'PatientSex'] = dcm_data.PatientSex

        data_df.loc[idx,'BodyPartExamined'] = dcm_data.BodyPartExamined

        data_df.loc[idx,'ViewPosition'] = dcm_data.ViewPosition

        

    return data_df
dicom_df = process_dicom_data(dicom_df)
# converting PatientAge to int as it is in float

dicom_df = dicom_df.astype({"PatientAge": int})

dicom_df.fillna(0.0, inplace=True)

dicom_df.head()
dicom_df.nunique()
plt.figure(figsize = (30, 10))

sns.countplot(x = 'PatientAge', hue = 'Target', data = dicom_df)
sns.countplot(x = 'PatientSex', hue = 'Target', data = dicom_df)
sns.countplot(x = 'ViewPosition', hue = 'Target', data = dicom_df);
dicom_df = dicom_df.drop('Target', axis=1)

dicom_df['PatientSex'].astype('category')

dicom_df['ViewPosition'].astype('category')

dicom_df['PatientSex'] = np.where(dicom_df["PatientSex"].str.contains("M"), 1, 0)

dicom_df['ViewPosition'] = np.where(dicom_df["ViewPosition"].str.contains("AP"), 1, 0)
dicom_df.head()
dicom_df.corr()
def show_dicom_image(data_df):

        img_data = list(data_df.T.to_dict().values())

        f, ax = plt.subplots(2,2, figsize=(16,18))

        for i,data_row in enumerate(img_data):

            pid = data_row['patientId']

            dcm_file = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % pid

            dcm_data = pydicom.read_file(dcm_file)                    

            ax[i//2, i%2].imshow(dcm_data.pixel_array, cmap=plt.cm.bone)

            ax[i//2, i%2].set_title('ID: {}\n Age: {} Sex: {}'.format(

                data_row['patientId'],dcm_data.PatientAge, dcm_data.PatientSex))
show_dicom_image(df[df['Target']==1].sample(n=4))
show_dicom_image(df[ (df['Target']==0) & (df['class']=='No Lung Opacity / Not Normal')].sample(n=4))
show_dicom_image(df[ (df['Target']==0) & (df['class']=='Normal')].sample(n=4))
def show_dicome_with_boundingbox(data_df):

    img_data = list(data_df.T.to_dict().values())

    f, ax = plt.subplots(2,2, figsize=(16,18))

    for i,data_row in enumerate(img_data):

        pid = data_row['patientId']

        dcm_file = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % pid

        dcm_data = pydicom.read_file(dcm_file)                    

        ax[i//2, i%2].imshow(dcm_data.pixel_array, cmap=plt.cm.bone)

        ax[i//2, i%2].set_title('ID: {}\n Age: {} Sex: {}'.format(

                data_row['patientId'],dcm_data.PatientAge, dcm_data.PatientSex))

        rows = data_df[data_df['patientId']==data_row['patientId']]

        box_data = list(rows.T.to_dict().values())        

        for j, row in enumerate(box_data):            

            x,y,width,height = row['x'], row['y'],row['width'],row['height']

            rectangle = Rectangle(xy=(x,y),width=width, height=height, color="red",alpha = 0.1)

            ax[i//2, i%2].add_patch(rectangle)            
show_dicome_with_boundingbox(df[df['Target']==1].sample(n=4))
from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers.core import Flatten, Dense, Dropout

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator
IMAGE_SIZE = [224, 224]



train_path = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/'

test_path = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images/'