# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import tqdm



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import matplotlib.pyplot as plt

%matplotlib inline
import pydicom

import glob

import os

from typing import Dict, List
def visualize_osic_images(image_files: List[str]) -> None:

    # Take only the first 12 images in the list

    image_files = image_files[:12]

    

    fig, axes = plt.subplots(4, 3, figsize=(20, 16))

    axes = axes.flatten()

    for image_index, image_file in enumerate(image_files):

        # Load the DICOM image and convert to pixel array

        image_data = pydicom.read_file(image_file).pixel_array

        axes[image_index].imshow(image_data, cmap=plt.cm.bone)

        

        image_name = '-'.join(image_file.split('/')[-2:])

        axes[image_index].set_title(f'{image_name}')
train_image_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/train'

train_image_files = glob.glob(os.path.join(train_image_path, '*', '*.dcm'))
visualize_osic_images(train_image_files)
image_data = pydicom.read_file(train_image_files[0])

image_data
# Different calls in the image import

image_data.PatientName, image_data.Modality, image_data.BodyPartExamined
train_df = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')

train_df.head()
# Counts of the Weeks field. We will look at the top 20 weeks contained in the train.csv



weeks_frequency = train_df['Weeks'].value_counts().head(20)

weeks_frequency = weeks_frequency.reset_index()

weeks_frequency = weeks_frequency.rename(columns={'index': 'Weeks', 'Weeks': 'Frequency'})



plt.figure(figsize=(10, 7))

ax = sns.barplot(x='Weeks', y='Frequency', data=weeks_frequency, order=weeks_frequency['Weeks'])

ax.set_title('Top Weeks by Frequency')

plt.grid()
# Histogram of the Age field



plt.figure(figsize=(10, 7))

ax = sns.distplot(train_df['Age'])

ax.set_title('Histogram for Age')

plt.grid()



print(train_df['Age'].describe())
sex_frequency = train_df['Sex'].value_counts()

sex_frequency = sex_frequency.reset_index()

sex_frequency = sex_frequency.rename(columns={'index': 'Sex', 'Sex': 'Frequency'})



plt.figure(figsize=(10, 7))

ax = sns.barplot(x='Sex', y='Frequency', data=sex_frequency, order=sex_frequency['Sex'])

ax.set_title('Sex Barplot')

plt.grid()
smoking_status_frequency = train_df['SmokingStatus'].value_counts()

smoking_status_frequency = smoking_status_frequency.reset_index()

smoking_status_frequency = smoking_status_frequency.rename(columns={'index': 'SmokingStatus', 'SmokingStatus': 'Frequency'})



plt.figure(figsize=(10, 7))

ax = sns.barplot(x='SmokingStatus', y='Frequency', data=smoking_status_frequency, order=smoking_status_frequency['SmokingStatus'])

ax.set_title('Smoking Status Barplot')

plt.grid()
# Histogram of the FVC field



plt.figure(figsize=(10, 7))

ax = sns.distplot(train_df['FVC'])

ax.set_title('Histogram for FVC')

plt.grid()



print(train_df['FVC'].describe())
# Histogram of the FVC field



plt.figure(figsize=(10, 7))

ax = sns.distplot(train_df['Percent'])

ax.set_title('Histogram for Percent')

plt.grid()



print(train_df['Percent'].describe())
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

axes = axes.flatten()



# Distribution for Age

age_male = train_df.loc[train_df['Sex'] == 'Male']['Age']

age_female = train_df.loc[train_df['Sex'] == 'Female']['Age']



sns.kdeplot(age_male, label='Male', shade=True, ax=axes[0])

sns.kdeplot(age_female, label='Female', shade=True, ax=axes[0])



axes[0].legend()

axes[0].set_title('Age Distribution by Sex')

axes[0].set_xlabel('Age')

axes[0].grid()



# Distribution for Smoking Status

age_ex_smoker = train_df.loc[train_df['SmokingStatus'] == 'Ex-smoker']['Age']

age_never_smoked = train_df.loc[train_df['SmokingStatus'] == 'Never smoked']['Age']

age_currently_smoking = train_df.loc[train_df['SmokingStatus'] == 'Currently smokes']['Age']



sns.kdeplot(age_ex_smoker, label='Ex-Smoker', shade=True, ax=axes[1])

sns.kdeplot(age_never_smoked, label='Never Smoked', shade=True, ax=axes[1])

sns.kdeplot(age_currently_smoking, label='Currently Smokes', shade=True, ax=axes[1])



axes[1].legend()

axes[1].set_title('Age Distribution by Smoking Status')

axes[1].set_xlabel('Age')

axes[1].grid()
sns.lmplot(x='Age', y='FVC', hue='Sex', col='Sex', data=train_df)
sns.lmplot(x='Age', y='FVC', hue='SmokingStatus', col='SmokingStatus', data=train_df)
sns.lmplot(x='Age', y='Percent', hue='Sex', col='Sex', data=train_df)
sns.lmplot(x='Age', y='Percent', hue='SmokingStatus', col='SmokingStatus', data=train_df)
sns.lmplot(x='Age', y='Weeks', hue='Sex', col='Sex', data=train_df)
sns.lmplot(x='Age', y='Weeks', hue='SmokingStatus', col='SmokingStatus', data=train_df)
image_data
def extract_dicom_meta_data(filename: str) -> Dict:

    # Load image

    image_data = pydicom.read_file(train_image_files[0])

    

    row = {

        'Patient': image_data.PatientID,

        'body_part_examined': image_data.BodyPartExamined,

        'image_position_patient': image_data.ImagePositionPatient,

        'image_orientation_patient': image_data.ImageOrientationPatient,

        'photometric_interpretation': image_data.PhotometricInterpretation,

        'rows': image_data.Rows,

        'columns': image_data.Columns,

        'pixel_spacing': image_data.PixelSpacing,

        'window_center': image_data.WindowCenter,

        'window_width': image_data.WindowWidth

    }

    

    return row
meta_data_df = []

for filename in tqdm.tqdm(train_image_files):

    meta_data_df.append(extract_dicom_meta_data(filename))
# Convert to a pd.DataFrame from dict

meta_data_df = pd.DataFrame.from_dict(meta_data_df)

meta_data_df.head()