from IPython.display import IFrame, YouTubeVideo

YouTubeVideo('cRVRAKM5ono',width=600, height=400)
!pip install fastai2 -q
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



#Load the dependancies

from fastai2.basics import *

from fastai2.callback.all import *

from fastai2.vision.all import *

from fastai2.medical.imaging import *



import pydicom
df_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

df_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
df_train.head()
df_train.shape,df_test.shape
df_train.nunique()
df_test.nunique()
df_weeks = df_train.groupby("Patient").agg({"Weeks":"nunique","Age":"nunique"}).reset_index()

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10, 5))

sns.countplot(df_weeks.Weeks,ax = ax1);

sns.countplot(df_weeks.Age,ax =ax2);
df_patients = df_train[["Patient","Sex","SmokingStatus","Age"]].drop_duplicates()

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20, 5),gridspec_kw={'width_ratios': [1,1,2]})

sns.countplot(df_patients.Sex,ax = ax1);

sns.countplot(df_patients.SmokingStatus,ax =ax2);

sns.countplot(df_patients.Age,ax =ax3);
sns.lineplot(x = "Weeks", y = "FVC", data = df_train[df_train.Patient=="ID00007637202177411956430"]);
TRAIN_DATA = "../input/osic-pulmonary-fibrosis-progression/train"
train_files = get_dicom_files(TRAIN_DATA)

train_files
info_view = train_files[0]

dimg = dcmread(info_view)

dimg
dimg.show()