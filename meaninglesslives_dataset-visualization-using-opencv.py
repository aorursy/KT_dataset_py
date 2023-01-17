import os
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from skimage.transform import resize
from skimage.morphology import label
from skimage.feature import hog
from skimage import exposure
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.feature import canny
from skimage.filters import sobel
from skimage.morphology import watershed
from scipy import ndimage as ndi
import warnings
warnings.filterwarnings("ignore")
from skimage.segmentation import mark_boundaries
from scipy import signal
import cv2
import glob, pylab, pandas as pd
import pydicom, numpy as np
import tqdm
import gc
# gc.enable()
# https://www.kaggle.com/peterchang77/exploratory-data-analysis
def parse_data(df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': '../input/stage_1_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed
# https://www.kaggle.com/peterchang77/exploratory-data-analysis
def draw(data,im):
    """
    Method to draw single patient with bounding box(es) if present 

    """

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)
        
    return im

def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im
df = pd.read_csv('../input/stage_1_train_labels.csv')
parsed = parse_data(df)
df.head()
det_class_path = '../input/stage_1_detailed_class_info.csv'
det_class_df = pd.read_csv(det_class_path)
det_class_df.head()
import cv2
from IPython.display import display, Image
def cvshow(image, format='.png', rate=255 ):
    decoded_bytes = cv2.imencode(format, image*rate)[1].tobytes()
    display(Image(data=decoded_bytes))
    return
j = 0
df = det_class_df[det_class_df['class']=='No Lung Opacity / Not Normal']
# nImg = df.shape[0]/3  # takes long time to load !!!
nImg = 400
img_ar = np.empty(0)
df = df.reset_index()
while img_ar.shape[0]!=nImg:
# for j in range(nImg):
    ind = np.random.randint(df.shape[0])
    patientId = df['patientId'][ind]
    dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    img = np.expand_dims(dcm_data.pixel_array,axis=0)    
    if j==0:
        img_ar = img
    elif (j%100==0):
        print(j,'images loaded')
    else:
        img_ar = np.concatenate([img_ar,img],axis=0)
    j += 1
    
def imgtile(imgs,tile_w):
    assert imgs.shape[0]%tile_w==0,"'imgs' cannot divide by 'th'."
    r=imgs.reshape((-1,tile_w)+imgs.shape[1:])
    return np.hstack(np.hstack(r))

#usage
tiled = imgtile(img_ar,20)
# cvshow(tiled)
tiled.shape
cvshow(cv2.resize( tiled, (1024,1024), interpolation=cv2.INTER_LINEAR ))
from ipywidgets import interact,IntSlider
@interact
def f(i=IntSlider(min=1,max=18,step=1,value=0)):
    cvshow(imgtile(img_ar[i*20:(i+1)*20],5))
j = 0
df = det_class_df[det_class_df['class']=='Normal']
# nImg = df.shape[0]/3  # takes long time to load !!!
nImg = 400
df = df.reset_index()
img_ar = np.empty(0)
while img_ar.shape[0]!=nImg:
# for j in range(nImg):
    ind = np.random.randint(df.shape[0])
    patientId = df['patientId'][ind]
    dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    img = np.expand_dims(dcm_data.pixel_array,axis=0)    
    if j==0:
        img_ar = img
    elif (j%100==0):
        print(j,'images loaded')
    else:
        img_ar = np.concatenate([img_ar,img],axis=0)
    j += 1
    
#usage
tiled = imgtile(img_ar,20)
tiled.shape
cvshow(cv2.resize( tiled, (1024,1024), interpolation=cv2.INTER_LINEAR ))
j = 0
df = det_class_df[det_class_df['class']=='Lung Opacity']
# nImg = df.shape[0]/3  # takes long time to load !!!
nImg = 400
img_ar = np.empty(0)
df = df.reset_index()
img_ar = np.empty(0)
while img_ar.shape[0]!=nImg:
# for j in range(nImg):
    ind = np.random.randint(df.shape[0])
    patientId = df['patientId'][ind]
    dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    img = dcm_data.pixel_array
    data = parsed[patientId]
    img = draw(data,img)
    img = np.expand_dims(img,axis=0)    
    if j==0:
        img_ar = img
    elif (j%100==0):
        print(j,'images loaded')
    else:
        img_ar = np.concatenate([img_ar,img],axis=0)
    j += 1
#usage
tiled = imgtile(img_ar,20)
tiled.shape
cvshow(cv2.resize( tiled, (1024,1024) ))