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
df = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
df['Sex'] = df['Sex'].apply(lambda x : 1 if x=='Male' else 0)
df['SmokingStatus'] = df['SmokingStatus'].apply(lambda x : 0 if x=='Never smoked' else (1 if x=='Currently smokes'  else 2 ))
len(df.Patient.unique()),df.shape,df.drop_duplicates().shape
df.groupby(['Patient']).Weeks.nunique().describe()
df.groupby(['SmokingStatus']).Patient.nunique()
df.groupby(['Age']).Patient.nunique(()).plot()
df[df.SmokingStatus==0].groupby(['Age']).Patient.nunique(()).plot(label='ex-smoker')
df[df.SmokingStatus==1].groupby(['Age']).Patient.nunique(()).plot(label='currently smokes')
df[df.SmokingStatus==2].groupby(['Age']).Patient.nunique(()).plot(label='never smoked')
df.groupby(['Patient'],as_index=False).Percent.min().plot()
df.groupby(['Patient'],as_index=False).Percent.max().plot()
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import glob
%matplotlib inline

patient_ls = df.Patient.unique()
for i in range(10):
    data_sub = df[df.Patient==patient_ls[i]]
    min_week = data_sub[data_sub.Percent == data_sub.Percent.min()].Weeks.iloc[0]
    max_week = data_sub[data_sub.Percent == data_sub.Percent.max()].Weeks.iloc[0]
    if min_week<=0:    min_week=1
    elif max_week<=0:  max_week=1
    images = glob.glob(f"/kaggle/input/osic-pulmonary-fibrosis-progression/train/{patient_ls[i]}/*.dcm")
    week_num = [file for file in images if int(file.split('/')[-1].split(".")[0]) in [min_week,max_week]]
    try :
        plt.figure(figsize=(15,5))
        for idx,file in enumerate(week_num):
            plt.subplot(1,3,idx+1)
            img = pydicom.dcmread(file)
            plt.title(file.split('/')[-1].split(".")[0])
            plt.imshow(img.pixel_array)


        plt.subplot(1,3,3)
        plt.plot('Weeks','Percent',marker='o',data=data_sub)
        plt.title(patient_ls[i])
    except :
        pass


df.head(9)
images = glob.glob(f"/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/*.dcm")
set([int(file.split('/')[-1].split(".")[0]) for file in images])
img = pydicom.dcmread("/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/1.dcm")
# test_data = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
sub_data = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
sub_data['Patient'] = sub_data.Patient_Week.apply(lambda x: x.split("_")[0])
sub_data['Weeks'] = sub_data.Patient_Week.apply(lambda x: x.split("_")[1]).astype(int)
list(sub_data[sub_data.Patient == 'ID00419637202311204720264'].Weeks)
set([int(file.split('/')[-1].split(".")[0]) for file in images])
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import pydicom
%matplotlib inline

import torch
import torch.nn as nn
from torch.nn import Sequential

from skimage.segmentation import clear_border
from skimage.measure import regionprops,label
from skimage import segmentation,measure
import scipy.ndimage as ndimage
train_data = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
test_data = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
train_images = glob.glob("/kaggle/input/osic-pulmonary-fibrosis-progression/train/*/*")
train_images[:3]
def load_scan(images_ls):
    slices = [pydicom.read_file(file) for file in images_ls]
    slices.sort(key = lambda x: x.InstanceNumber)
    
    try :
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-slices[1].ImagePositionPatient[2])
    except :
        slice_thickness = np.abs(slices[0].SliceLocation-slices[1].SliceLocation)
        
    for s in slices:
        s.SliceWidth = slice_thickness
    
    return slices
        
def image_hu(slices):
    img_array = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    slope = slices[0].RescaleSlope
    intercept = slices[0].RescaleIntercept
    img_array[img_array==-2000] = 0
    
    img_array  = slope*img_array + intercept
    return img_array.astype(np.int16)

def creater_markers(img):
    internal1 = img<-400
    internal2 = segmentation.clear_border(internal1)
    internal_labels = measure.label(internal2)
    internal_labels1 = internal_labels.copy()
    
    areas =[r.area for r in measure.regionprops(internal_labels)]
    areas.sort()
    if len(areas)>2:
        for region in measure.regionprops(internal_labels):
            if region.area<areas[-2]:
                for coord in region.coords:
                    internal_labels[coord[0],coord[1]] = 0
    
    internal_labels = internal_labels>0
    
    external_a = ndimage.binary_dilation(internal_labels,iterations=10)
    external_b = ndimage.binary_dilation(internal_labels,iterations=55)
    external = external_b^external_a
    
    watershed = np.zeros((512,512),dtype=np.int)
    watershed1 = watershed + internal_labels*255
    watershed2 = watershed1 + external*128
        
    return internal_labels,external,watershed2
    
    
    
oneimage = load_scan(train_images[:2])
plt.figure(figsize=(15,5))
out_ls = creater_markers(image_hu(oneimage)[0])

plt.subplot(1,len(out_ls)+1,1)
plt.imshow(image_hu(oneimage)[0],cmap='gray')
for i in range(len(out_ls)):
    try :
        plt.subplot(1,len(out_ls)+1,i+2)
        plt.imshow(out_ls[i],cmap='gray')
    except : pass
