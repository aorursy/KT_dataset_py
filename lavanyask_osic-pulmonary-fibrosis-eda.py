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
os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression')
import cv2

import seaborn as sns

import matplotlib.pyplot as plt

import pydicom

import glob



%matplotlib inline
traindf = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

traindf.shape
traindf.head()
traindf.info()
testdf = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

testdf.shape
testdf.head()
testdf.info()
ROOT_DIR = '/kaggle/input/osic-pulmonary-fibrosis-progression/'

TRAIN_DIR = ROOT_DIR + 'train'

TEST_DIR = ROOT_DIR + 'test'
# getting a brief summary on all the values of train set

traindf.describe()
sns.boxplot(x = 'FVC', data = traindf)
sns.violinplot(x = 'FVC', data = traindf)
# correlation matrix

corrMatrix = traindf.corr()

print(corrMatrix)

# plotting a heatmap

sns.heatmap(corrMatrix, vmin = -1, vmax = 1, center = 0, cmap = 'BuGn');
traindf.Patient.value_counts()
# getting the number of unique Patient IDs

print('The number of unique patient IDs in train set:',traindf.Patient.nunique())
traindfgrouped = traindf.groupby(['Patient','Age','Sex','SmokingStatus']).agg({'Patient': ['count']})

traindfgrouped.columns = ['Patient_record_count']

traindfgrouped = traindfgrouped.reset_index()

print(traindfgrouped)
traindfgrouped.Patient_record_count.describe()
sns.set_style('darkgrid')

plt.figure(figsize=(15,8))

ax = sns.countplot(x = 'Age', data = traindfgrouped)

# number of unique age values

total = float(len(ax.patches))

for p in ax.patches:

    ht = p.get_height()

    ax.text(p.get_x(), ht+0.3, '{:1.2f}'.format(ht/total))
df = traindfgrouped

ag = df.groupby(['Age','Sex']).sum().unstack()

ag.columns = ag.columns.droplevel()

ag.plot(kind = 'bar', width = 1, colormap = 'Accent', figsize = (15,8))

plt.show()
ax = sns.countplot(x = 'Sex', data = traindfgrouped, palette = 'pastel')

# number of unique patients

total = 176.0

for p in ax.patches:

    ht = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.0, ht+0.3, '{:1.2f}%'.format(ht*100/total), ha = 'center')
sns.countplot(x = 'SmokingStatus', data = traindfgrouped, palette = 'pastel')
sns.countplot(x = 'SmokingStatus', hue = 'Sex', data = traindfgrouped, palette = 'pastel')
traindfgrouped[traindfgrouped['SmokingStatus']=='Currently smokes']
# plotting the CT scan of patient ID00060637202187965290703 for week 107

filepath1 = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00060637202187965290703/107.dcm'

file1 = pydicom.read_file(filepath1)

plt.imshow(file1.pixel_array, cmap = plt.cm.bone)

plt.title('Patient: ID00060637202187965290703')
patients = os.listdir(TRAIN_DIR)

patients.sort()
def load_scan(path):

    slices = [pydicom.read_file(os.path.join(path,s)) for s in os.listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices
def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)   

    return np.array(image, dtype=np.int16)
first_patient = load_scan(os.path.join(TRAIN_DIR, patients[0]))

first_patient_pixels = get_pixels_hu(first_patient)

plt.hist(first_patient_pixels.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()



# Show some slice in the middle

plt.title('Slice 20')

plt.imshow(first_patient_pixels[20], cmap=plt.cm.bone)

plt.show()
print(patients[0])
plt.figure(figsize = (18,18))

for i in range(30):

    plt.subplot(5,6,i+1)

    plt.imshow(first_patient_pixels[i],cmap=plt.cm.bone)

    plt.title('slice ' + str(i+1))