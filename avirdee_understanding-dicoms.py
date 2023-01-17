!pip install fastai2 -q
#Load the dependancies
from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *
from fastai2.medical.imaging import *

import pydicom
import seaborn as sns

import numpy as np
import pandas as pd
import os

sns.set(style="whitegrid")
sns.set_context("paper")
source = Path('../input/osic-pulmonary-fibrosis-progression')
files = os.listdir(source)
print(files)
train = source/'train'
train_files = get_dicom_files(train)
train_files
info_view = train_files[777]
dimg = dcmread(info_view)
dimg
dimg.PixelData[:200]
dimg.pixel_array, dimg.pixel_array.shape
dimg.show()
px = dimg.pixels.flatten()
plt.hist(px, color='c')
tensor_dicom = pixels(dimg) #convert into tensor

print(f'RescaleIntercept: {dimg.RescaleIntercept:1f}\nRescaleSlope: {dimg.RescaleSlope:1f}\nMax pixel: '
      f'{tensor_dicom.max()}\nMin pixel: {tensor_dicom.min()}\nShape: {tensor_dicom.shape}')
tensor_dicom_scaled = scaled_px(dimg) #convert into tensor taking RescaleIntercept and RescaleSlope into consideration
plt.hist(tensor_dicom_scaled.flatten(), color='c')
print(f'Max pixel: {tensor_dicom_scaled.max()}\nMin pixel: {tensor_dicom_scaled.min()}')
dimg.show(max_px=None, min_px=300, figsize=(7,7))
dimg.show(max_px=100, min_px=-100, figsize=(7,7))
dimg.show(max_px=None, min_px=-1000, figsize=(7,7))
dimg.show(max_px=-1000, min_px=-2000, figsize=(7,7))
def show_one_image(file):
    """ function to view patient image and choosen tags within the head of the DICOM"""
    pat = dcmread(file)
    print(f'Patient ID: {pat.PatientID}')
    print(f'File Number: {pat.InstanceNumber}')
    print(f'\nWindow Center: {pat.WindowCenter}')
    print(f'Window Width: {pat.WindowWidth}')
    print(f'Rescale Intercept: {pat.RescaleIntercept}')
    print(f'Rescale Slope: {pat.RescaleSlope}')
    print(f'Body part: {pat.BodyPartExamined}')
    img = dcmread(file)
    return img.show()
show_one_image(info_view)
df = pd.read_csv(source/'train.csv')
df.head()
#Plot 3 comparisons
def plot_comparison3(df, feature, feature1, feature2):
    "Plot 3 comparisons from a dataframe"
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (25, 4))
    s1 = sns.countplot(df[feature], ax=ax1)
    s1.set_title(feature)
    s2 = sns.countplot(df[feature1], ax=ax2)
    s2.set_title(feature1)
    s3 = sns.countplot(df[feature2], ax=ax3)
    s3.set_title(feature2)
    plt.show()
plot_comparison3(df, 'Age', 'Sex', 'SmokingStatus')
#Plot 1 comparisons
def plot_comparison1(df, feature):
    "Plot 1 comparisons from a dataframe"
    fig, (ax1) = plt.subplots(1,1, figsize = (25, 4))
    s1 = sns.countplot(df[feature], ax=ax1)
    s1.set_title(feature)
    plt.show()
plot_comparison1(df, 'Patient')
no_of_patients = df.Patient.nunique()
no_of_patients
patient_01 = df[df.Patient == 'ID00082637202201836229724']
patient_01
patient_01.plot(kind='line',x='Weeks',y='FVC',color='red')
plt.show()
patient_01.plot(kind='line',x='Weeks',y='Percent', color='c')
plt.show()
patient001 = train/'ID00082637202201836229724'
patient001_files = get_dicom_files(patient001)
patient001_files
@delegates(subplots)
def show_images(ims, nrows=1, ncols=None, titles=None, cmap=None, **kwargs):
    "Show all images `ims` as subplots with `rows` using `titles`"
    if ncols is None: ncols = int(math.ceil(len(ims)/nrows))
    if titles is None: titles = [None]*len(ims)
    axs = subplots(nrows, ncols, **kwargs)[1].flat
    for im,t,ax in zip(ims, titles, axs): show_image(im, ax=ax, title=t, cmap=cmap)
im_list = []
def get_files(pat):
        folder = f'{train}/{pat}'
        fl = get_dicom_files(folder)
        for file in fl:
            im_list.append(file)
get_files('ID00082637202201836229724')
sorted_list = sorted(im_list, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
pat_list =[]
def view_patient(l):
    for file in l:
        trans = Transform(Resize(256))
        dicom_create = PILDicom.create(file)
        dicom_transform = trans(dicom_create)
        pat_list.append(dicom_transform)
    show_images(pat_list, cmap='bone', nrows=38)
view_patient(sorted_list)