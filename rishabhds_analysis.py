import pandas as pd

from matplotlib import pyplot as plt

import pydicom

import os

import random

import numpy as np
train = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
train.head()
len(train['Patient'].unique())
plt.hist(train.groupby(['Patient']).count()['Weeks'].to_list())

plt.show()
import matplotlib.pyplot as plt

#fig = plt.Figure(figsize=(28,42),constrained_layout=True)

fig, axs = plt.subplots(2, 2,figsize=(15,10))

axs[0, 0].hist(train['Age'])

axs[0, 0].set_title('Age Distribution')

axs[0, 1].hist(train['SmokingStatus'])

axs[0, 1].set_title('Smokers Distribution')

axs[1, 0].hist(train['Sex'])

axs[1, 0].set_title('Gender Distribution')

axs[1, 1].hist(train['Weeks'],bins=50)

axs[1, 1].set_title('Weeks Distribution')

plt.show()
width = (max(train['Weeks'])-min(train['Weeks']))/4

row = 2

col = 2

_min = min(train['Weeks'])

data = dict()

for i in range(0,row):

    for j in range(0,col):

        c_graph = i*col+j

        llimit = _min+c_graph*width

        ulimit = llimit+width

        key = 'Weeks '+str(llimit)+' - '+str(ulimit)

        data[key] = train[(train['Weeks']<=ulimit) & (train['Weeks']>llimit)]['Percent']

fig, ax = plt.subplots(figsize=(15,10))

ax.boxplot(data.values())

ax.set_title('Distibution of percent with Week #')

ax.set_xticklabels(data.keys())

plt.show()
width = (max(train['Age'])-min(train['Age']))/4

row = 2

col = 2

_min = min(train['Age'])

data = dict()

for i in range(0,row):

    for j in range(0,col):

        c_graph = i*col+j

        llimit = _min+c_graph*width

        ulimit = llimit+width

        key = 'Age '+str(llimit)+' - '+str(ulimit)

        data[key] = train[(train['Age']<=ulimit) & (train['Age']>llimit)]['FVC']

fig, ax = plt.subplots(figsize=(15,10))

ax.boxplot(data.values())

ax.set_title('Distibution of FVC with Patient\'s age')

ax.set_xticklabels(data.keys())

plt.show()
width = (max(train['Age'])-min(train['Age']))/4

row = 2

col = 2

_min = min(train['Age'])

data = dict()

for i in range(0,row):

    for j in range(0,col):

        c_graph = i*col+j

        llimit = _min+c_graph*width

        ulimit = llimit+width

        key = 'Age '+str(llimit)+' - '+str(ulimit)

        data[key] = train[(train['Age']<=ulimit) & (train['Age']>llimit)]['Percent']

fig, ax = plt.subplots(figsize=(15,10))

ax.boxplot(data.values())

ax.set_title('Distibution of percent with Patient\'s Age')

ax.set_xticklabels(data.keys())

plt.show()
train['SmokingStatus'].unique()
data = dict()

smokingstatus = train['SmokingStatus'].unique()

for i in range(len(smokingstatus)):

    key = smokingstatus[i]

    data[key] = train[train['SmokingStatus']==key]['Percent']

fig, ax = plt.subplots(figsize=(15,10))

ax.boxplot(data.values())

ax.set_title('Distibution of percent with Patient\'s Smoking Status')

ax.set_xticklabels(data.keys())

plt.show()
data = dict()

genders = train['Sex'].unique()

for i in range(len(genders)):

    key = genders[i]

    data[key] = train[train['Sex']==key]['Percent']

fig, ax = plt.subplots(figsize=(15,10))

ax.boxplot(data.values())

ax.set_title('Distibution of percent with Patient\'s Gender')

ax.set_xticklabels(data.keys())

plt.show()
keys = ['BitsAllocated','BitsStored',

 'BodyPartExamined',

 'Columns',

 'ConvolutionKernel',

 'DeidentificationMethod',

 'DistanceSourceToDetector',

 'DistanceSourceToPatient',

 'FocalSpots',

 'FrameOfReferenceUID',

 'GantryDetectorTilt',

 'GeneratorPower',

 'HighBit',

 'ImageOrientationPatient',

 'ImagePositionPatient',

 'ImageType',

 'InstanceNumber',

 'KVP',

 'Manufacturer',

 'ManufacturerModelName',

 'Modality',

 'PatientID',

 'PatientName',

 'PatientPosition',

 'PatientSex',

 'PhotometricInterpretation',

 'PixelData',

 'PixelRepresentation',

 'PixelSpacing',

 'PositionReferenceIndicator',

 'RescaleIntercept',

 'RescaleSlope',

 'RotationDirection',

 'Rows',

 'SOPInstanceUID',

 'SamplesPerPixel',

 'SeriesInstanceUID',

 'SliceLocation',

 'SliceThickness',

 'StudyID',

 'StudyInstanceUID',

 'TableHeight',

 'WindowCenter',

 'WindowCenterWidthExplanation',

 'WindowWidth',

 'XRayTubeCurrent']
all_image_meta = dict()

from datetime import datetime as dt

from tqdm import tqdm

for key in keys:

    all_image_meta[key] = list()

Break = False

for _file in tqdm(os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/train/')):

    for _img in os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/train/'+_file):

        dataset = pydicom.dcmread('/kaggle/input/osic-pulmonary-fibrosis-progression/train/'+_file+'/'+_img)

        all_image_meta['file_path'] = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/'+ _file + '/' + _img

        for key in keys:

            try:

                all_image_meta[key].append(str(dataset[key]).split(' ')[-1])

            except KeyError as k:

                all_image_meta[key].append(None)

            except Exception as e:

                Break = True

                break

        if Break:

            break

    if Break:

        break
dataset

all_image_meta = pd.DataFrame(all_image_meta)

all_image_meta.shape
for key in keys:

    print(key,all_image_meta[key].nunique(),all_image_meta[key][0])
def clean_patient_id(_str):

    return _str.replace('\'','')
all_image_meta['PatientID'] = all_image_meta['PatientID'].apply(clean_patient_id)
all_image_meta.groupby('PatientID').count()['Columns'].mean()
def view_patient(patient_id):

    # Draw random 15 images for the patient

    files = os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/train/'+patient_id)

    files = random.sample(files, 15)

    

    # Draw the images

    row = 3

    col = 5

    fig, axs = plt.subplots(row, col,figsize=(15,10))

    for i in range(0,row):

        for j in range(0,col):

            c_image = files[i*col+j]

            data = pydicom.dcmread('/kaggle/input/osic-pulmonary-fibrosis-progression/train/'+patient_id+'/'+c_image)

            axs[i, j].imshow(data.pixel_array, cmap=plt.cm.bone)

            axs[i,j].set_title(c_image)

    plt.show()

    

    # plot the FVC and percentage progression for the patient

    data = train[train['Patient'] == patient_id]

    plt.plot(data['Weeks'], data['FVC'],color = 'red')

    plt.show()
patient_id = np.random.choice(train[(train['Percent']>80) & (train['Age'] < 55)]['Patient'].to_list())

view_patient(patient_id)
patient_id = np.random.choice(train[(train['Percent']>80) & (train['Age'] > 55)]['Patient'].to_list())

view_patient(patient_id)
patient_id = np.random.choice(train[(train['Percent']<60) & (train['Age'] < 55)]['Patient'].to_list())

view_patient(patient_id)
patient_id = np.random.choice(train[(train['Percent']<60) & (train['Age'] > 55)]['Patient'].to_list())

view_patient(patient_id)
patient_id = np.random.choice(train[(train['SmokingStatus'] == 'Currently smokes')]['Patient'].to_list())

view_patient(patient_id)
patient_id = np.random.choice(train[(train['SmokingStatus'] == 'Never smoked')]['Patient'].to_list())

view_patient(patient_id)
patient_id = np.random.choice(train[(train['SmokingStatus'] == 'Ex-smoker')]['Patient'].to_list())

view_patient(patient_id)