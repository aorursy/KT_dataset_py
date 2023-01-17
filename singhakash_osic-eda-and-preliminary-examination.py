# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install chart_studio
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# import plotly.plotly as py
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


sns.set()
plt.rcParams["figure.figsize"] = [8,12]


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv(r"/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
df_test=pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
df_train.head(170)
df_test.head()
print("The shape of our training dataset is:{}".format(df_train.shape))
print("The shape of our testing dataset is:{}".format(df_test.shape))
df_train.info()
df_test.info()
df_train.describe()
df_train['Patient'].nunique()
df_unique = df_train.groupby([df_train.Patient,df_train.Age,df_train.Sex, df_train.SmokingStatus])['Patient'].count()
df_unique.index = df_unique.index.set_names(['Patient Id','Age','Sex','SmokingStatus'])

df_unique = df_unique.reset_index()
df_unique.rename(columns = {'Patient': 'Frequency'},inplace = True)

df_unique.head()
df_unique['Frequency'].mean()
print(df_unique['Sex'].value_counts())
percent_sexwise=df_unique['Sex'].value_counts()/len(df_unique['Sex'])
print(percent_sexwise)
fig, ax = plt.subplots(figsize=(20,8))
sns.countplot(ax=ax, x="Sex", data=df_unique, palette="bone")
from matplotlib import colors

fig, ax = plt.subplots(figsize=(20,8))
N_points = 100000
n_bins = 15
N, bins, patches = plt.hist(df_unique['Age'], n_bins, alpha=0.5)

fracs = N / N.max()
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
fig, ax = plt.subplots(figsize=(20,8))
df_train.groupby(['Age', 'Sex']).size().unstack().plot(kind='bar', stacked=True, ax=ax)
fig, ax = plt.subplots(figsize=(20,8))
sns.countplot(x="SmokingStatus", data=df_unique, palette="magma")
fig, ax = plt.subplots(figsize=(20,8))
df_train.groupby(['SmokingStatus', 'Sex']).size().unstack().plot(kind='bar', stacked=True, ax=ax)
fig, ax = plt.subplots(figsize=(20,8))

N_points = 100000
n_bins = 15
N, bins, patches = plt.hist(df_train['FVC'], n_bins, alpha=0.5)

fracs = N / N.max()
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.inferno(norm(thisfrac))
    thispatch.set_facecolor(color)
print(df_train['FVC'].min())
print(df_train['FVC'].max())
print(df_train['FVC'].mean())
print(df_train['FVC'].median())
from scipy.stats import kurtosis, skew


print("The mean of FVC data is: {}".format(df_train['FVC'].mean()))
print("The median of FVC data is: {}".format(df_train['FVC'].median()))
print("The standard deviation of FVC data is: {}".format(df_train['FVC'].std()))

print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(df_train['FVC']) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(df_train['FVC']) ))
fig, ax = plt.subplots(figsize=(20,15))
sns.boxplot(x="Sex", y="FVC",hue="SmokingStatus", data=df_train, palette="spring", ax=ax)
fig, ax = plt.subplots(figsize=(20,8))
df_weeks=df_train.groupby(['Weeks']).count()
df_weeks.head(20)
sns.distplot(df_weeks, color='b')
fig, ax = plt.subplots(figsize=(20,15))

x=df_train['FVC']
y=df_train['Percent']

colors = df_train['FVC']  # 0 to 15 point radii

plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()
df_patient=df_train.groupby(['Patient'])
df_patient.head()
patient_df = df_train[df_train['Patient'] == "ID00010637202177584971671"]
print(patient_df)
import plotly.express as px

fig = px.line(patient_df, x="Weeks", y="FVC", title='Patient FVC over the weeks')
fig.show()
patient_df = df_train[df_train['Patient'] == "ID00426637202313170790466"]
print(patient_df)
import plotly.express as px

fig = px.line(patient_df, x="Weeks", y="FVC", title='Patient FVC over the weeks')
fig.show()
patient_df = df_train[df_train['Patient'] == "ID00082637202201836229724"]
print(patient_df)
import plotly.express as px

fig = px.line(patient_df, x="Weeks", y="FVC", title='Patient FVC over the weeks')
fig.show()
# common packages 
import numpy as np 
import os
import copy
from math import *
import matplotlib.pyplot as plt
from functools import reduce
# reading in dicom files
import pydicom as dicom
import glob
# skimage image processing packages
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops
# scipy linear algebra functions 
from scipy.linalg import norm
import scipy.ndimage
# ipywidgets for some interactive plots
from ipywidgets.widgets import * 
import ipywidgets as widgets
# plotly 3D interactive graphs 
import plotly
from plotly.graph_objs import *
import chart_studio.plotly as py
# set plotly credentials here 
# this allows you to send results to your account plotly.tools.set_credentials_file(username=your_username, api_key=your_key)
apply_resample = False
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
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
def set_lungwin(img, hu=[-1200., 600.]):
    lungwin = np.array(hu)
    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg
scans = load_scan('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')
scan_array = set_lungwin(get_pixels_hu(scans))
from scipy.ndimage.interpolation import zoom

def resample(imgs, spacing, new_spacing):
    new_shape = np.round(imgs.shape * spacing / new_spacing)
    true_spacing = spacing * imgs.shape / new_shape
    resize_factor = new_shape / imgs.shape
    imgs = zoom(imgs, resize_factor, mode='nearest')
    return imgs, true_spacing, new_shape

spacing_z = (scans[-1].ImagePositionPatient[2] - scans[0].ImagePositionPatient[2]) / len(scans)

if apply_resample:
    scan_array_resample = resample(scan_array, np.array(np.array([spacing_z, *scans[0].PixelSpacing])), np.array([1.,1.,1.]))[0]
import imageio
from IPython.display import Image

imageio.mimsave("/tmp/gif.gif", scan_array, duration=0.0001)
Image(filename="/tmp/gif.gif", format='png')