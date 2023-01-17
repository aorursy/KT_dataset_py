import os

import glob



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import torch

import torchvision



import pydicom



#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device:', DEVICE)

DATA_PATH = '/kaggle/input/osic-pulmonary-fibrosis-progression/'
scan_nums_per_patient = {os.path.split(dirname)[-1]:len(filenames)

                         for dirname, _, filenames

                         in os.walk(DATA_PATH + 'train/')}

scan_nums_per_patient = pd.Series(tuple(scan_nums_per_patient.values()),

                                  index=scan_nums_per_patient.keys())

_, ax = plt.subplots(figsize=(10, 8))

scan_nums_per_patient.plot(kind='hist', bins=30, ax=ax);
# each patient has their own directory containing their

# entire history of DICOM scan sets



# list of all patients available in the training set

patients = glob.glob(DATA_PATH + 'train/*')



# pick a patient at random from the above list

# with >0 shots

sample_patient = np.random.choice(patients[1:])



# now we need a list of all the paths to their scans

patient_scans = glob.glob(sample_patient + '/*')



# pick one of this patient's scans at random

sample_scan = np.random.choice(patient_scans)

print(f"Patient: {os.path.split(sample_patient)[-1]}")

print(f"Scan: {os.path.split(sample_scan)[-1]}")



# `pydicom.dcmread` will read a `.dcm` file

# using the correct image encoding

# it returns a `DataSet` iterable containing

# one image slice and all of the patient's information

dataset = pydicom.dcmread(sample_scan)



# the `dir` keyword lets us explore the attributes,

# class methods and instance methods of a particular object:

#dir(dataset)
type(dataset)
dataset
{data_elem.keyword: data_elem.value for data_elem in dataset.values() if not data_elem.keyword=='PixelData'}
plt.imshow(dataset.pixel_array, cmap=plt.cm.bone);
# sort our sample patient's scan by the number in the filename

patient_scans_by_number = sorted(patient_scans,

                                 key=lambda x: 

                                 int(os.path.splitext(os.path.split(x)[-1])[0]))



# choose nine equally-spaced sequential images from this patient's set of scans

idx = np.round(np.linspace(0, len(patient_scans_by_number) - 1, 9)).astype(int)

# subset the scan images with the indices we made above

sample_scan2 = [patient_scans_by_number[i] for i in idx]



# read each of the nine images with `pydicom`

sample_sets = [pydicom.dcmread(image) for image in sample_scan2]

# extract the pixel array from each of the nine images

images = [image.pixel_array for image in sample_sets]



# tricky thing here:

# so far i've noticed that most of these images are encoded as floats

# but there are some edge cases encoded as `uint16`s, which are too big

# for `torch` to construct a `torch.Tensor` out of.

# if we happen upon one of those cases, we need to convert the image

# array to `uint8` so that `torch` doesn't throw a `TypeError`



# we use `torchvision.transforms.ToTensor` because if that edge case arises,

# it will handle the logic required to scale a [0,255] RGB image into 

# the [0,1] RGB format desired by `plt`

try:

    images_tensor = [torchvision.transforms.ToTensor()(image) for image in images]

except TypeError:

    images_tensor = [torchvision.transforms.ToTensor()(image.astype('uint8')) for image in images]



# use `make_grid` to arrange a 3x3 visualization of the entire chest scan

# add some padding so that the grid isn't so crowded

grid = torchvision.utils.make_grid(images_tensor, nrow=3, padding=100)
# create some axes with a nice, big `figsize` so we can look closely

_, ax = plt.subplots(figsize=(12,10))



# permute the grid tensor so it has the image number as the last channel

# finally, show the image grid with a pretty CT scan color palette!

ax.imshow(grid.permute((1,2,0)), cmap=plt.cm.bone);
# verifying that all of these are indeed part of a chest scan

print('All samples are chest scans:',

      all([sample['BodyPartExamined'].value == 'Chest' for sample in sample_sets]))
# verifying that all of these are indeed part of a chest scan

print('All samples used the LUNG kernel for reconstruction:',

      all([sample['ConvolutionKernel'].value in ('LUNG', 'L') for sample in sample_sets]))