import pandas as pd

import numpy as np

import os

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt

import pandas_profiling 

from pandas_profiling import ProfileReport 
df_train=pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
ProfileReport(df_train)
df_train.shape
df_train.info()
df_train.head()
#df_train['Age'].hist()

#import seaborn as sns

age=df_train['Age']

age=pd.Series(age,name="Age")

ax=sns.distplot(age)
df_train['Sex'].hist()

#A big percentage of people in this df are males as males are at higher risks as compared to women
df_train['SmokingStatus'].hist()

#A big percentage of people in this df are ex smoker and never smoked are also at risk
df_train.describe()
df_train.isnull().sum()

#Train is complete with all details
df_test=pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
ProfileReport(df_test)
df_test.head()
df_test.info()
df_test.describe()
df_test.shape
age=df_test['Age']

age=pd.Series(age,name="Age")

ax=sns.distplot(age)
df_test['Sex'].hist()
df_test['SmokingStatus'].hist()
#Let's see what sample submission looks like 

sub=pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
ProfileReport(sub)
sub.head()
sub.shape
sub.info
sub.describe
#Import pydicom

#Pydicom is a python package specifically for parsing .dcm files

import pydicom

from pydicom.data import get_testdata_files

preview="../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/10.dcm"

metadata=pydicom.dcmread(preview)

metadata
PixelDims = (int(metadata.Rows), int(metadata.Columns), len(preview))

print(PixelDims)
print("Patient id....................:", metadata.PatientID)

print("Modality......................:", metadata.Modality)

print("BodyPartExamined..............:", metadata.BodyPartExamined)  

print("Image Position    (Patient)...:", metadata.ImagePositionPatient)

print("Image Orientation (Patient)...:", metadata.ImageOrientationPatient)
plt.imshow(metadata.pixel_array, cmap=plt.cm.bone)

plt.show()
patient = df_train[df_train.Patient == 'ID00007637202177411956430']

patient
#Let's see how fvc change over weeks

patient.plot(kind='line',x='Weeks',y='FVC',color='red')

plt.show()
patient.plot(kind='line',x='Weeks',y='Percent', color='c')

plt.show()