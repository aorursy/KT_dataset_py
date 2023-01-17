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




# ------------------------------------Loading the csv data==================

import pandas as pd
Detailed_Class_info = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')
Train_Labels = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')

# viewing detailed Class Info

Detailed_Class_info.head()
# print(Detailed_Class_info.count))
print (Detailed_Class_info['class'].unique)
# Viewing train labels
Train_Labels.head()
# print(Train_Labels.count())

# 0 - No Pneumonia detected
# 1 - Pneumonia detected with the co-ordinates of bounding box
#Removing Duplicates from both sets and joining the dataframes on patient_ID for better data visualization

Detailed_Class_info = Detailed_Class_info.drop_duplicates('patientId').reset_index(drop=True)

Train_Labels = Train_Labels.drop_duplicates('patientId').reset_index(drop=True)

Data=Train_Labels.merge(Detailed_Class_info, how='inner', on='patientId')

Data.head(15)


# Let's try to visualize our target vairable counts

import seaborn as sns
sns.countplot(x="class",hue="class",data=Detailed_Class_info)

# Approx 6k patients with Pneumonia detected
# 12k approx have no lung opacity but classified as no pneumonia
# 8k-10k normal cases
# Checking if the class imbalance is observed

sns.countplot(x="Target",hue="Target",data=Train_Labels)
# Looking at the counts below data imbalance is found in 0 and 1 classes
# We have more data with No Pneumonia detected cases
# Visualizing images from the dataset
import pydicom as dcm
from pydicom import dcmread
#Copying all files with .dcm extension into a list of train images

import glob 
Train_image_list= glob.glob('../input/rsna-pneumonia-detection-challenge/stage_2_train_images/*.dcm')
# Reading all the image data from the list of dcm files
Train_Images = [dcm.read_file(x, stop_before_pixels=False) for x in Train_image_list]
# Checking the data type of the image 

# Its a DICOM image with tags written to it along with image
type(Train_Images[1])
# Let's try to print image data
print(Train_Images[1])

# Some imoortant tags can be seen from the data like Patient_ID ,Patient_Name
import pylab
import numpy as np
# Visualizing the images with patient id's in as X labels

fig=pylab.figure()

fig.set_size_inches(15,5)

fig1=fig.add_subplot(1,4,1)

fig1.set_xlabel(Train_Images[0].PatientID)

fig2=fig.add_subplot(1,4,2)

fig2.set_xlabel(Train_Images[1].PatientID)

fig3=fig.add_subplot(1,4,3)

fig3.set_xlabel(Train_Images[2].PatientID)

fig4=fig.add_subplot(1,4,4)

fig4.set_xlabel(Train_Images[3].PatientID)

fig1.imshow(Train_Images[0].pixel_array, cmap=pylab.cm.bone)

fig2.imshow(Train_Images[1].pixel_array, cmap=pylab.cm.bone)

fig3.imshow(Train_Images[2].pixel_array, cmap=pylab.cm.bone)

fig4.imshow(Train_Images[3].pixel_array, cmap=pylab.cm.bone)

Patient1=Data[Data['patientId']==Train_Images[0].PatientID]
Patient2=Data[Data['patientId']==Train_Images[1].PatientID]
Patient3=Data[Data['patientId']==Train_Images[2].PatientID]
Patient4=Data[Data['patientId']==Train_Images[3].PatientID]

print("Patient 1---->",Patient1['Target'])
print("Patient 2---->",Patient2['Target'])
print("Patient 3---->",Patient3['Target'])
print("Patient 4---->",Patient4['Target'])
# From the images above its evident that if lung opacity is obhserved its like to have pneumonia.
Patient3
# Visualizing Image with bounding box over the affected area as per the data

import matplotlib.patches as patches

fig=pylab.figure()

fig.set_size_inches(15,5)

fig1=fig.add_subplot(1,1,1)

fig1.imshow(Train_Images[2].pixel_array, cmap=pylab.cm.bone)

rect = patches.Rectangle((321,246),285,525, edgecolor='r', facecolor="none")

fig1.add_patch(rect)
# Let's capture lables and extract pixel data from images
 

print(Data.shape)
print(len(Train_Images))

# Checking if the data in csv and images are in same sequence

print(Data.head(1))
print(Train_Images[0].PatientID)

# Images are not in the same sequence