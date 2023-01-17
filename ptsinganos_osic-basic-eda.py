# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import ipympl

import matplotlib.pyplot as plt

import matplotlib

import pydicom

import pandas_profiling as pdp



%matplotlib inline
TRAIN_CSV_PATH = '/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv'

TEST_CSV_PATH = '/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv'



TRAIN_DICOM_DIR = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/'

TEST_DICOM_DIR = '/kaggle/input/osic-pulmonary-fibrosis-progression/test/'

df = pd.read_csv(TRAIN_CSV_PATH)

df.head(4)
# Unique patients

df['Patient'].nunique()
df['Sex'].value_counts().plot(kind='bar')
# Histogram of Age

df.groupby('Sex')['Age'].plot(kind='hist', legend=True, alpha=0.5)
# Histogram of measurements/patient

df['Patient'].value_counts().plot(kind='hist', bins=np.arange(df['Patient'].value_counts().max()+2))
# Histogram of FVC values

df['FVC'].plot(kind='hist')
df.groupby( ['Sex','SmokingStatus'] )['FVC'].agg( ['mean','std','count'] )
i = df.Patient == 'ID00007637202177411956430'

df[i][['Weeks', 'FVC']].plot(kind='line', x='Weeks', y='FVC', style=['d-'])

plt.title(f'Patient {df[i]["Patient"][0]}:\n{df[i]["Sex"][0]} / {df[i]["Age"][0]} / {df[i]["SmokingStatus"][0]}')

plt.show()
# Generate report (https://www.kaggle.com/piantic/osic-pulmonary-fibrosis-progression-basic-eda)

report = pdp.ProfileReport(df)
report
series = np.array(

    [

        [

            (os.path.join(dp, f), pydicom.dcmread(os.path.join(dp, f), stop_before_pixels = True))

            for f in files

        ]

        for dp,_,files in os.walk(TRAIN_DICOM_DIR) if len(files) != 0

    ]

)

series[0][0][1]
# Total files

instances = [f for l in series for f in l]

len(instances)
# Patients

patient_ids = np.unique([inst[1].PatientID for inst in instances])

len(patient_ids)
# Series (3D volumes)

len(series)
# How many studies?

studies = {}



for s in series:

    studies.setdefault(s[0][1].StudyInstanceUID, []).append(s)



len(studies)
# Studies per patient

[len([st for st in studies.values() if st[0][0][1].PatientID == p]) for p in patient_ids]
# Series per study

series_per_study = [(len(sr), sr[0][0][1].PatientID) for sr in studies.values()]

series_per_study
# Images per series

img_per_series = [len(s) for s in series]

print(img_per_series)
res = {}

spc = {}

thck = {}



for sr in series:

    try:

        sr.sort(key = lambda inst: int(inst[1].ImagePositionPatient[2]))

    except:

        sr.sort(key = lambda inst: int(inst[1].InstanceNumber))

    

    dcm = sr[0][1]

    key = str(dcm.PixelSpacing)

    spc.setdefault(key, [])

    spc[key].append((dcm.PatientID, ))#dcm.StudyDescription, dcm.StudyDate, dcm.SeriesDescription))

    

    key = str((dcm.Rows, dcm.Columns))

    res.setdefault(key, [])

    res[key].append((dcm.PatientID, ))#dcm.StudyDescription, dcm.StudyDate, dcm.SeriesDescription))

    

    try:

        key = str(np.abs(sr[0][1].ImagePositionPatient[2] - sr[1][1].ImagePositionPatient[2]))

        thck.setdefault(key, [])

        thck[key].append((dcm.PatientID, ))#dcm.StudyDescription, dcm.StudyDate, dcm.SeriesDescription))

    except:

        print(dcm.PatientID)

        continue

    

thck.keys()
res.keys()
spc.keys()
res
for key in res.keys():

    print(f'{key}:{len(res[key])}')
for key in ['(1302, 1302)', '(843, 888)', '(632, 632)', '(1100, 888)', '(788, 888)', '(734, 888)', '(752, 888)', '(733, 888)']:

    print(f'{key}: {res[key]}')
spc
for key in spc.keys():

    print(f'{key}:{len(spc[key])}')
seq1 = r"ID00210637202257228694086"

seq1_slices = [pydicom.dcmread(os.path.join(TRAIN_DICOM_DIR, seq1, f)) for f in os.listdir(os.path.join(TRAIN_DICOM_DIR, seq1))]

seq1_slices.sort(key = lambda inst: int(inst.ImagePositionPatient[2]))



seq2 = r"ID00132637202222178761324"

seq2_slices = [pydicom.dcmread(os.path.join(TRAIN_DICOM_DIR, seq2, f)) for f in os.listdir(os.path.join(TRAIN_DICOM_DIR, seq2))]

try:

    seq2_slices.sort(key = lambda inst: int(inst.ImagePositionPatient[2]))

except:

    seq2_slices.sort(key = lambda inst: int(inst.InstanceNumber))
seq1 = np.stack([s.pixel_array for s in seq1_slices])

seq2 = np.stack([s.pixel_array for s in seq2_slices])
[ipp.ImagePositionPatient for ipp in seq1_slices]
seq1.shape, seq2.shape
plt.imshow(seq1[150,:,:], cmap='gray')
plt.imshow(seq2[150,:,:], cmap='gray')