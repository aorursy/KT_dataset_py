from IPython.display import HTML

HTML('<center><iframe width="560" height="315" src="https://www.youtube.com/embed/8UnPPZlnfbk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pydicom as dicom

import matplotlib.pyplot as plt

from os import listdir,mkdir

import plotly.express as px

import seaborn as sns

import os
basepath = "../input/rsna-str-pulmonary-embolism-detection/"

listdir(basepath)
train = pd.read_csv("/kaggle/input/rsna-str-pulmonary-embolism-detection/train.csv")

test = pd.read_csv("/kaggle/input/rsna-str-pulmonary-embolism-detection/test.csv")

sub = pd.read_csv("/kaggle/input/rsna-str-pulmonary-embolism-detection/sample_submission.csv")
print("Training Data Size")

train.shape
print("Test Data Size")

test.shape
train.head(10)
train.tail(10)
test.head(10)
test.tail(10)
sub.head(10)
sub.tail(10)
train.info()
train.describe()
test.info()
sub.info()
print('Check missing value in train data')

train.isnull().sum()
print('Chack missing value in test data')

test.isnull().sum()
x = train.drop(['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], axis=1).sum(axis=0).sort_values().reset_index()

x.columns = ['column', 'nonzero_records']



fig = px.bar(

    x, 

    x='nonzero_records', 

    y='column', 

    orientation='h', 

    title='Columns and non zero samples', 

    height=800, 

    width=800

)



fig.show()

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

def set_lungwin(img, hu=[-1200., 600.]):

    lungwin = np.array(hu)

    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])

    newimg[newimg < 0] = 0

    newimg[newimg > 1] = 1

    newimg = (newimg * 255).astype('uint8')

    return newimg

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
scans = load_scan('../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf/')

scan_array = set_lungwin(get_pixels_hu(scans))
import matplotlib.animation as animation



fig = plt.figure()



ims = []

for image in scan_array:

    im = plt.imshow(image, animated=True, cmap="Greys")

    plt.axis("off")

    ims.append([im])



ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,

                                repeat_delay=1000)

HTML(ani.to_jshtml())

def transform_to_hu(slices):

    images = np.stack([file.pixel_array for file in slices])

    images = images.astype(np.int16)



    # convert ouside pixel-values to air:

    # I'm using <= -1000 to be sure that other defaults are captured as well

    images[images <= -1000] = 0

    

    # convert to HU

    for n in range(len(slices)):

        

        intercept = slices[n].RescaleIntercept

        slope = slices[n].RescaleSlope

        

        if slope != 1:

            images[n] = slope * images[n].astype(np.float64)

            images[n] = images[n].astype(np.int16)

            

        images[n] += np.int16(intercept)

    

    return np.array(images, dtype=np.int16)



def load_slice(path):

    slices = [dicom.read_file(path + '/' + s) for s in listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices

first_patient = load_slice('../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf')

first_patient_pixels = transform_to_hu(first_patient)



def sample_stack(stack, rows=6, cols=6, start_with=0, show_every=5):

    fig,ax = plt.subplots(rows,cols,figsize=[18,20])

    for i in range(rows*cols):

        ind = start_with + i*show_every

        ax[int(i/rows),int(i % rows)].set_title(f'slice {ind}')

        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='bone')

        ax[int(i/rows),int(i % rows)].axis('off')

    plt.show()



sample_stack(first_patient_pixels)
cols = train.copy()

cols.drop(['StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID'],axis=1,inplace=True)

corr = cols.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(12, 12))

    ax = sns.heatmap(corr,mask=mask,square=True,linewidths=.8,cmap="summer",annot=True)
