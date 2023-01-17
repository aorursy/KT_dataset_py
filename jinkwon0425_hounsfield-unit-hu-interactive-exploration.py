import os
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
%matplotlib inline

# dcm 
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

from colorama import Fore, Back, Style


# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# Settings for pretty nice plots
plt.style.use('fivethirtyeight')
plt.show()

from ipywidgets import interact, interactive, IntSlider, Layout
import ipywidgets as widgets

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
random.seed(12)

# take a random sample of a training DICOM file
TRAIN_FILE_PATH = "/kaggle/input/rsna-str-pulmonary-embolism-detection/train"
sample_StudyInstanceUID = random.choice(os.listdir(TRAIN_FILE_PATH))

FULL_SAMPLE_PATH = TRAIN_FILE_PATH + '/' + sample_StudyInstanceUID
sample_SeriesInstanceUID = random.choice(os.listdir(FULL_SAMPLE_PATH))

FULL_SAMPLE_PATH += '/' + sample_SeriesInstanceUID
sample_SOPInstanceUID = os.listdir(FULL_SAMPLE_PATH)[0]

FULL_SAMPLE_PATH += '/' + sample_SOPInstanceUID
print("Selected Sample: " + FULL_SAMPLE_PATH)
# get dicom instance
img_dicom = pydicom.read_file(FULL_SAMPLE_PATH)
sample_img = img_dicom.pixel_array

# apply rescaling
slope = img_dicom.RescaleSlope
intercept = img_dicom.RescaleIntercept
sample_img = sample_img * slope + intercept
# static plots
DIM = 10

fig= plt.figure(figsize=(DIM,DIM))
W = 400
L = 40

img_min = L - W // 2 # lowest_visible_value
img_max = L + W // 2 # highest_visible_value

test_img = sample_img.copy()

test_img[test_img < img_min] = img_min # setting any value lower than lowest_visible_value to lowest_visible_value 
test_img[test_img > img_max] = img_max # setting any value higher than highest_visible_value to highest_visible_value

plt.imshow(test_img)
plt.show()
DIM = 6 

min_hu = min(sample_img.flatten().tolist())//10*10
max_hu = max(sample_img.flatten().tolist())//10*10

W = widgets.IntSlider(description='W', min = min_hu, max = max_hu, step = 10)
L = widgets.IntSlider(description='L', min = min_hu, max = max_hu, step = 10)

def f(W, L):
    img_min = L - W // 2 # lowest_visible_value
    img_max = L + W // 2 # highest_visible_value

    test_img = sample_img.copy()

    test_img[test_img < img_min] = img_min # setting any value lower than lowest_visible_value to lowest_visible_value 
    test_img[test_img > img_max] = img_max # setting any value higher than highest_visible_value to highest_visible_value
    fig = plt.figure(figsize=(DIM,DIM))
    plt.imshow(test_img)

out = widgets.interactive_output(f, {'W': W, 'L': L})

widgets.HBox([widgets.VBox([W, L]), out])