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
det_class_path = '../input/stage_1_detailed_class_info.csv'
det_class_df = pd.read_csv(det_class_path)
det_class_df.head()
# simple features that can be easily extracted and used for training deep networks
# these features may be used along with original image

plt.figure(figsize=(30,15))
# plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0)  #adjust this to change vertical and horiz. spacings..
nImg = 3  #no. of images to process
j = -1
df = det_class_df[det_class_df['class']=='No Lung Opacity / Not Normal']
df = df.reset_index()
while True:
# for j in range(nImg):
    if j == nImg-1:
        break
        
    ind = np.random.randint(df.shape[0])
    patientId = df['patientId'][ind]
    dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    img = dcm_data.pixel_array
    
    data = parsed[patientId]
    j += 1
        
    q = j+1
    
#     # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    
    # Equalization
    img_eq = exposure.equalize_hist(img)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img)

    plt.subplot(nImg,5,q*5-4)
    plt.imshow(draw(data,img), cmap='binary')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(nImg,5,q*5-3)    
    plt.imshow(draw(data,img_rescale), cmap='binary')
    plt.title('Contrast stretching')
    plt.axis('off')
    
    plt.subplot(nImg,5,q*5-2)
    plt.imshow(draw(data,img_eq), cmap='binary')
    plt.title('Equalization')
    plt.axis('off')
    
    plt.subplot(nImg,5,q*5-1)
    plt.imshow(draw(data,img_adapteq), cmap='binary')
    plt.title('Adaptive Equalization')
    plt.axis('off')

plt.show()
plt.tight_layout()
# simple features that can be easily extracted and used for training deep networks
# these features may be used along with original image

plt.figure(figsize=(30,15))
# plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0)  #adjust this to change vertical and horiz. spacings..
nImg = 3  #no. of images to process
j = -1
df = det_class_df[det_class_df['class']=='Normal']
df = df.reset_index()
while True:
# for j in range(nImg):
    if j == nImg-1:
        break
        
    ind = np.random.randint(df.shape[0])
    patientId = df['patientId'][ind]
    dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    img = dcm_data.pixel_array
    
    data = parsed[patientId]
    j += 1
        
    q = j+1
    
#     # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    
    # Equalization
    img_eq = exposure.equalize_hist(img)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img)

    plt.subplot(nImg,5,q*5-4)
    plt.imshow(draw(data,img), cmap='binary')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(nImg,5,q*5-3)    
    plt.imshow(draw(data,img_rescale), cmap='binary')
    plt.title('Contrast stretching')
    plt.axis('off')
    
    plt.subplot(nImg,5,q*5-2)
    plt.imshow(draw(data,img_eq), cmap='binary')
    plt.title('Equalization')
    plt.axis('off')
    
    plt.subplot(nImg,5,q*5-1)
    plt.imshow(draw(data,img_adapteq), cmap='binary')
    plt.title('Adaptive Equalization')
    plt.axis('off')

plt.show()
plt.tight_layout()
# simple features that can be easily extracted and used for training deep networks
# these features may be used along with original image

plt.figure(figsize=(30,15))
# plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0)  #adjust this to change vertical and horiz. spacings..
nImg = 3  #no. of images to process
j = -1
df = det_class_df[det_class_df['class']=='Lung Opacity']
df = df.reset_index()
while True:
# for j in range(nImg):
    if j == nImg-1:
        break
        
    ind = np.random.randint(df.shape[0])
    patientId = df['patientId'][ind]
    dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    img = dcm_data.pixel_array
    
    data = parsed[patientId]
    j += 1
        
    q = j+1
    
#     # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    
    # Equalization
    img_eq = exposure.equalize_hist(img)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img)

    plt.subplot(nImg,5,q*5-4)
    plt.imshow(draw(data,img), cmap='binary')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(nImg,5,q*5-3)    
    plt.imshow(draw(data,img_rescale), cmap='binary')
    plt.title('Contrast stretching')
    plt.axis('off')
    
    plt.subplot(nImg,5,q*5-2)
    plt.imshow(draw(data,img_eq), cmap='binary')
    plt.title('Equalization')
    plt.axis('off')
    
    plt.subplot(nImg,5,q*5-1)
    plt.imshow(draw(data,img_adapteq), cmap='binary')
    plt.title('Adaptive Equalization')
    plt.axis('off')

plt.show()
plt.tight_layout()
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
model = VGG16(weights='imagenet', include_top=False)
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
# vgg features that can be easily extracted and used for training deep networks
# these features may be used along with original image
random.seed(40)
plt.figure(figsize=(30,15))
plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0.2)  #adjust this to change vertical and horiz. spacings..
nImg = 3  #no. of images to process
for j in range(nImg):
    q = j+1
    
    ind = np.random.randint(df.shape[0])
    patientId = df['patientId'][ind]
    dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    img = dcm_data.pixel_array
    img = cv2.resize(img,(224, 224))
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img,3,axis=2)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    
    layer_outs = [func([x, 0.]) for func in functors]
    feat = np.reshape(layer_outs[4][0],(112,112,128))
    layer4 = np.max(feat,axis=2)
    
    feat = np.reshape(layer_outs[6][0],(56,56,128))
    layer6 = np.max(feat,axis=2)
    
    feat = np.reshape(layer_outs[10][0],(28,28,256))
    layer10 = np.max(feat,axis=2)
    
    plt.subplot(nImg,6,q*6-5)
    plt.imshow(img, cmap='binary')
    plt.title('Original Image')
    
    plt.subplot(nImg,6,q*6-4)
    plt.imshow(img, cmap='binary')
    plt.title('Image Mask')
    
    plt.subplot(nImg,6,q*6-3)    
    plt.imshow(layer4, cmap='binary')
    plt.title('VGG Layer 4')
    
    plt.subplot(nImg,6,q*6-2)
    plt.imshow(layer6, cmap='binary')
    plt.title('VGG Layer 6')
    
    plt.subplot(nImg,6,q*6-1)
    plt.imshow(layer10, cmap='binary')
    plt.title('VGG Layer 10')


plt.show()
# model.summary()
model = ResNet50(weights='imagenet',input_shape=(224, 224, 3), include_top=False)
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
# model.summary()
# resnet features that can be easily extracted and used for training deep networks
# these features may be used along with original image
random.seed(40)
plt.figure(figsize=(15,30))
# plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0.2)  #adjust this to change vertical and horiz. spacings..
nImg = 5  #no. of images to process
for j in range(nImg):
    q = j+1
    
    ind = np.random.randint(df.shape[0])
    patientId = df['patientId'][ind]
    dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    img = dcm_data.pixel_array
    img = cv2.resize(img,(224, 224))
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img,3,axis=2)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    
    layer_outs = [func([x, 0.]) for func in functors]
    feat = np.reshape(layer_outs[4][0],(112,112,64))
    layer4 = np.max(feat,axis=2)
    
    plt.subplot(nImg,3,q*3-2)
    plt.imshow(img, cmap='binary')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(nImg,3,q*3-1)    
    plt.imshow(layer4, cmap='binary')
    plt.title('ResNet activation_1 ')
    plt.axis('off')


plt.show()
# plt.tight_layout()
model = Xception(weights='imagenet', include_top=False)
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
# Xception features that can be easily extracted and used for training deep networks
# these features may be used along with original image
random.seed(40)
plt.figure(figsize=(30,15))
plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0.2)  #adjust this to change vertical and horiz. spacings..
nImg = 3  #no. of images to process
for j in range(nImg):
    q = j+1
    
    ind = np.random.randint(df.shape[0])
    patientId = df['patientId'][ind]
    dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    img = dcm_data.pixel_array
    img = cv2.resize(img,(224, 224))
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img,3,axis=2)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    
    layer_outs = [func([x, 0.]) for func in functors]
    feat = np.reshape(layer_outs[4][0],(109,109,64))
    layer4 = np.max(feat,axis=2)
    
    feat = np.reshape(layer_outs[16][0],(55,55,128))
    layer6 = np.max(feat,axis=2)
    
    feat = np.reshape(layer_outs[26][0],(28,28,256))
    layer10 = np.max(feat,axis=2)
    
    plt.subplot(nImg,6,q*6-5)
    plt.imshow(img, cmap='binary')
    plt.title('Original Image')
    
    plt.subplot(nImg,6,q*6-4)
    plt.imshow(img, cmap='binary')
    plt.title('Image Mask')
    
    plt.subplot(nImg,6,q*6-3)    
    plt.imshow(layer4, cmap='binary')
    plt.title('Xception Block 1')
    
    plt.subplot(nImg,6,q*6-2)
    plt.imshow(layer6, cmap='binary')
    plt.title('Xception Block 2')
    
    plt.subplot(nImg,6,q*6-1)
    plt.imshow(layer10, cmap='binary')
    plt.title('Xception Block 3 ')


plt.show()