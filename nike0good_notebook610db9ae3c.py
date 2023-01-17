

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import copy

from datetime import timedelta, datetime

import imageio

import matplotlib.pyplot as plt

from matplotlib import cm

import multiprocessing

import numpy as np

import os

from pathlib import Path

import pydicom

import pytest

import scipy.ndimage as ndimage

from scipy.ndimage.interpolation import zoom

from skimage import measure, morphology, segmentation

from skimage.transform import resize

from time import time, sleep

from tqdm import trange, tqdm

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import *

from tensorflow.data import Dataset

import torch

from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

import warnings

import seaborn as sns

import glob as glob

import imageio

from IPython.display import Image



#for masking

from skimage.measure import label,regionprops

from sklearn.cluster import KMeans

from skimage.segmentation import clear_border



import onnx



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

print('{} Rows and {} Columns in train data '.format(train_df.shape[0], train_df.shape[1]))

train_df.head()
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

test_df.head()
data_path = '../input/osic-pulmonary-fibrosis-progression/train/'



output_path = '../input/output/'

train_image_files = sorted(glob.glob(os.path.join(data_path, '*','*.dcm')))

patients = os.listdir(data_path)

patients.sort()



print('Some sample Patient ID''s :', len(train_image_files))

print("\n".join(train_image_files[:5]))
def load_scan(path):

    """

    Loads scans from a folder and into a list.

    

    Parameters: path (Folder path)

    

    Returns: slices (List of slices)

    """

    

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices
def get_pixels_hu(scans):



    image = np.stack([s.pixel_array for s in scans])

    image = image.astype(np.int16)



    # Since the scanning equipment is cylindrical in nature and image output is square,

    # we set the out-of-scan pixels to 0

    image[image == -2000] = 0

    

    

    # HU = m*P + b

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)

        

    image += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)

test_patient_scans = load_scan(data_path + patients[2])

test_patient_images = get_pixels_hu(test_patient_scans)



#We'll be taking a random slice to perform segmentation:



for imgs in range(len(test_patient_images[0:5])):

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))

    ax1.imshow(test_patient_images[imgs], cmap=plt.cm.bone)

    ax1.set_title("Original Slice")

    

    ax2.imshow(test_patient_images[imgs], cmap=plt.cm.bone)

    ax2.set_title("Original Slice")

    

    ax3.imshow(test_patient_images[imgs], cmap=plt.cm.bone)

    ax3.set_title("Original Slice")

    plt.show()
def set_lungwin(img, hu=[-1200., 600.]):

    lungwin = np.array(hu)

    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])

    newimg[newimg < 0] = 0

    newimg[newimg > 1] = 1

    newimg = (newimg * 255).astype('uint8')

    return newimg





scans = load_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')

scan_array = set_lungwin(get_pixels_hu(scans))



imageio.mimsave("/tmp/gif.gif", scan_array, duration=0.0001)

Image(filename="/tmp/gif.gif", format='png')
def generate_markers(image):

    """

    Generates markers for a given image.

    

    Parameters: image

    

    Returns: Internal Marker, External Marker, Watershed Marker

    """

    

    #Creation of the internal Marker

    marker_internal = image < -400

    marker_internal = segmentation.clear_border(marker_internal)

    marker_internal_labels = measure.label(marker_internal)

    

    areas = [r.area for r in measure.regionprops(marker_internal_labels)]

    areas.sort()

    

    if len(areas) > 2:

        for region in measure.regionprops(marker_internal_labels):

            if region.area < areas[-2]:

                for coordinates in region.coords:                

                       marker_internal_labels[coordinates[0], coordinates[1]] = 0

    

    marker_internal = marker_internal_labels > 0

    

    # Creation of the External Marker

    external_a = ndimage.binary_dilation(marker_internal, iterations=10)

    external_b = ndimage.binary_dilation(marker_internal, iterations=55)

    marker_external = external_b ^ external_a

    

    # Creation of the Watershed Marker

    marker_watershed = np.zeros((512, 512), dtype=np.int)

    marker_watershed += marker_internal * 255

    marker_watershed += marker_external * 128

    

    return marker_internal, marker_external, marker_watershed
test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(test_patient_images[15])



f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))



ax1.imshow(test_patient_internal, cmap='gray')

ax1.set_title("Internal Marker")

ax1.axis('off')



ax2.imshow(test_patient_external, cmap='gray')

ax2.set_title("External Marker")

ax2.axis('off')



ax3.imshow(test_patient_watershed, cmap='gray')

ax3.set_title("Watershed Marker")

ax3.axis('off')



plt.show()
sample_image = pydicom.dcmread(train_image_files[7])

img = sample_image.pixel_array



plt.imshow(img, cmap='gray')

plt.title('Original Image')
img = (img + sample_image.RescaleIntercept) / sample_image.RescaleSlope

img = img < -400 #HU unit range for lungs CT SCAN



plt.imshow(img, cmap='gray')

plt.title('Binary Mask Image')
img = clear_border(img)

plt.imshow(img, cmap='gray')

plt.title('Cleaned Border Image')
img = label(img)

plt.imshow(img, cmap='gray')
areas = [r.area for r in regionprops(img)]

areas.sort()

if len(areas) > 2:

    for region in regionprops(img):

        if region.area < areas[-2]:

            for coordinates in region.coords:                

                img[coordinates[0], coordinates[1]] = 0

img = img > 0

plt.imshow(img, cmap='gray')
# https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/



def make_lungmask(img, display=False):

    row_size= img.shape[0]

    col_size = img.shape[1]

    

    mean = np.mean(img)

    std = np.std(img)

    img = img-mean

    img = img/std

    

    # Find the average pixel value near the lungs

        # to renormalize washed out images

    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 

    mean = np.mean(middle)  

    max = np.max(img)

    min = np.min(img)

    

    # To improve threshold finding, I'm moving the 

    # underflow and overflow on the pixel spectrum

    img[img==max]=mean

    img[img==min]=mean

    

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)

    

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))

    centers = sorted(kmeans.cluster_centers_.flatten())

    threshold = np.mean(centers)

    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image



    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  

    # We don't want to accidentally clip the lung.



    eroded = morphology.erosion(thresh_img,np.ones([3,3]))

    dilation = morphology.dilation(eroded,np.ones([8,8]))



    labels = measure.label(dilation) # Different labels are displayed in different colors

    label_vals = np.unique(labels)

    regions = measure.regionprops(labels)

    good_labels = []

    for prop in regions:

        B = prop.bbox

        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:

            good_labels.append(prop.label)

    mask = np.ndarray([row_size,col_size],dtype=np.int8)

    mask[:] = 0





    #  After just the lungs are left, we do another large dilation

    #  in order to fill in and out the lung mask 

    

    for N in good_labels:

        mask = mask + np.where(labels==N,1,0)

    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation



    if (display):

        fig, ax = plt.subplots(3, 2, figsize=[12, 12])

        ax[0, 0].set_title("Original")

        ax[0, 0].imshow(img, cmap='gray')

        ax[0, 0].axis('off')

        ax[0, 1].set_title("Threshold")

        ax[0, 1].imshow(thresh_img, cmap='gray')

        ax[0, 1].axis('off')

        ax[1, 0].set_title("After Erosion and Dilation")

        ax[1, 0].imshow(dilation, cmap='gray')

        ax[1, 0].axis('off')

        ax[1, 1].set_title("Color Labels")

        ax[1, 1].imshow(labels)

        ax[1, 1].axis('off')

        ax[2, 0].set_title("Final Mask")

        ax[2, 0].imshow(mask, cmap='gray')

        ax[2, 0].axis('off')

        ax[2, 1].set_title("Apply Mask on Original")

        ax[2, 1].imshow(mask*img, cmap='gray')

        ax[2, 1].axis('off')

        

        plt.show()

    return mask*img
# Select a sample

path = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/19.dcm"

dataset = pydicom.dcmread(path)

img = dataset.pixel_array



# Masked image

mask_img = make_lungmask(img, display=True)
import re

patient_dir = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430"

datasets = []



# First Order the files in the dataset

files = []

for dcm in list(os.listdir(patient_dir)):

    files.append(dcm) 

files.sort(key=lambda f: int(re.sub('\D', '', f)))



# Read in the Dataset

for dcm in files:

    path = patient_dir + "/" + dcm

    datasets.append(pydicom.dcmread(path))

    

imgs = []

for data in datasets:

    img = data.pixel_array

    imgs.append(img)

    

    

# Show masks

fig=plt.figure(figsize=(16, 6))

columns = 10

rows = 3



for i in range(1, columns*rows +1):

    img = make_lungmask(datasets[i-1].pixel_array)

    fig.add_subplot(rows, columns, i)

    plt.imshow(img, cmap="gray")

    plt.title(i, fontsize = 9)

    plt.axis('off');
def get_observation_data(path):

    '''Get information from the .dcm files.

    path: complete path to the .dcm file'''



    image_data = pydicom.read_file(path)



    # Dictionary to store the information from the image

    observation_data = {

        "PatientID" : image_data.PatientID,

        "SliceThickness" : int(image_data.SliceThickness),

        "KVP" : int(image_data.KVP),

        "DistanceSourceToDetector" : int(image_data.DistanceSourceToDetector),

        "DistanceSourceToPatient" : int(image_data.DistanceSourceToPatient),

        "GantryDetectorTilt" : int(image_data.GantryDetectorTilt),

        "TableHeight" : int(image_data.TableHeight),

        "XRayTubeCurrent" : int(image_data.XRayTubeCurrent),

        "GeneratorPower" : int(image_data.GeneratorPower),

      

        "WindowCenter" : int(image_data.WindowCenter),

        "WindowWidth" : int(image_data.WindowWidth),

        "PixelPaddingValue" : image_data.PixelPaddingValue,

        "SamplesPerPixel" : image_data.SamplesPerPixel,

        "SliceLocation" : int(image_data.SliceLocation),

        "BitsAllocated" : image_data.BitsAllocated,

        "BitsStored" : image_data.BitsStored,

        "HighBit" : image_data.HighBit,

        "PixelRepresentation" : image_data.PixelRepresentation,

        "RescaleIntercept" : int(image_data.RescaleIntercept),

        "RescaleSlope" : int(image_data.RescaleSlope),

    }

    

    return observation_data
meta_data_df = []

for filename in tqdm(train_image_files):

    try:

       meta_data_df.append(get_observation_data(filename))

    except Exception as e:

       continue
meta_data_df = pd.DataFrame.from_dict(meta_data_df)

meta_data_df
cols = [col for col in meta_data_df.columns if col not in['PatientID']]
md1=meta_data_df.groupby('PatientID').max()

md2=meta_data_df.groupby('PatientID').min()

md3=meta_data_df.groupby('PatientID').mean()

md1=meta_data_df.groupby('PatientID').max()

md4=pd.merge(md1,md2,on="PatientID",suffixes=("_1","_2"))

md5=pd.merge(md4,md3,on="PatientID",suffixes=("_o","_3"))

md5=md5.reset_index()


md5=md5.rename(columns={'PatientID':'Patient'})
md5
train = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/train.csv' )

test  = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/test.csv' )



train['traintest'] = 0

test ['traintest'] = 1



sub   = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/sample_submission.csv' )

sub['Weeks']   = sub['Patient_Week'].apply( lambda x: int(x.split('_')[-1]) )

sub['Patient'] = sub['Patient_Week'].apply( lambda x: x.split('_')[0] ) 

train.Patient.nunique(), sub.Patient.nunique()

sub.Patient.isin( test.Patient.unique() ).mean()

train = pd.concat( (train,test) )

train.sort_values( ['Patient','Weeks'], inplace=True )

train.shape

train.describe()



FE=[]

Ls=['Sex','SmokingStatus']

for col in Ls:

    for i in train[col].unique():

        FE.append(i)

        train[i] = (train[col] == i).astype(int)

        

train=train.drop(Ls,axis=1)
train
train=pd.merge(train,md5,on="Patient",how="left",suffixes=("_h","_c"))
train
train=train.fillna(train.mean())

train=train.fillna(train.mode())

train
train[i].isnull().sum()
def metric( trueFVC, predFVC, predSTD ):

    clipSTD = np.clip( predSTD, 70 , 9e9 )  

    deltaFVC = np.clip( np.abs(trueFVC-predFVC), 0 , 1000 ) 

    return np.mean( -1*(np.sqrt(2)*deltaFVC/clipSTD) - np.log( np.sqrt(2)*clipSTD ) )



dt = train.loc[ train.traintest==1]

dt

dt= dt.drop(['FVC','Weeks'],axis=1)
sub
test = pd.merge( sub, dt, on='Patient', how='left' )

test=test.drop(['FVC','Confidence'],axis=1)
test.sort_values( ['Patient','Weeks'], inplace=True )

test
import numpy as np

import pandas as pd

import pydicom

import os

import random

import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold

import tensorflow as tf

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")



def score(y_true, y_pred):

    tf.dtypes.cast(y_true, tf.float32)

    tf.dtypes.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]

    

    #sigma_clip = sigma + C1

    sigma_clip = tf.maximum(sigma, C1)

    delta = tf.abs(y_true[:, 0] - fvc_pred)

    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )

    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)

    return K.mean(metric)

#============================#

def qloss(y_true, y_pred):

    # Pinball loss for multiple quantiles

    qs = [0.2, 0.50, 0.8]

    q = tf.constant(np.array([qs]), dtype=tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    return K.mean(v)

#=============================#

def mloss(_lambda):

    def loss(y_true, y_pred):

        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)

    return loss

#=================

def make_model(nh):

    z = L.Input((nh,), name="Patient")

    x = L.Dense(100, activation="relu", name="d1")(z)

    x = L.Dense(100, activation="relu", name="d2")(x)

    p1 = L.Dense(3, activation="linear", name="p1")(x)

    p2 = L.Dense(3, activation="relu", name="p2")(x)

    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 

                     name="preds")([p1, p2])

    

    model = M.Model(z, preds, name="CNN")

    model.compile(loss=mloss(0.8), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])

    return model
def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    

seed_everything(42)

cols = train.columns
cols=cols.drop(['Patient','FVC'])

cols

train.columns
test.columns
y = train['FVC'].values.astype(np.float32)

z = train[cols].values.astype(np.float32)

nh = z.shape[1]

net = make_model(nh)
test


ze = test[cols].values.astype(np.float32)

pe = np.zeros((ze.shape[0], 3))

pred = np.zeros((z.shape[0], 3))
print(len(test))
NFOLD = 5

kf = KFold(n_splits=NFOLD)

BATCH_SIZE=128

cnt = 0

EPOCHS = 800

for tr_idx, val_idx in kf.split(z):

    cnt += 1

    print(f"FOLD {cnt}")

    net = make_model(nh)

    net.fit(z[tr_idx], y[tr_idx], batch_size=BATCH_SIZE, epochs=EPOCHS, 

            validation_data=(z[val_idx], y[val_idx]), verbose=0) #

    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))

    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))

    print("predict val...")

    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)

    print("predict test...")

    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD

len(pe)
test['FVC'] = 0.996*pe[:, 1]

test['Confidence'] = pe[:, 2] - pe[:, 0]


test[['Patient_Week','FVC','Confidence']].to_csv('submission.csv', index=False)

test