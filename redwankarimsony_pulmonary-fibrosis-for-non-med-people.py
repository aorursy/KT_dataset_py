from IPython.lib.display import YouTubeVideo

YouTubeVideo('AfK9LPNj-Zo', width=800, height=600)
import os

import json

from pathlib import Path

from glob import glob



import matplotlib.pyplot as plt

%matplotlib inline



train_data_dir = '../input/osic-pulmonary-fibrosis-progression/train/'

test_data_dir = '../input/osic-pulmonary-fibrosis-progression/test/'
patient_ids = os.listdir(train_data_dir)

print('Training Patient NOs:', len(patient_ids))



patient_ids_test = os.listdir(test_data_dir)

print('Test Patient NOs:', len(patient_ids_test))
train_image_paths_dcom  = glob(train_data_dir + '*/*.dcm')

print(f'Total train images {len(train_image_paths_dcom)}')
import pydicom

from pydicom.data import get_testdata_files

print(__doc__)
print(train_image_paths_dcom[0])

RefDs = pydicom.dcmread(train_image_paths_dcom[0])

print(f'Image size: {RefDs.Rows}x{RefDs.Columns}' )

RefDs
from ipywidgets import interact, interactive, IntSlider, ToggleButtons

def explore_patients_metadata(patient_id, instance):

    RefDs = pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/' + 

                            patient_id +'/' + 

                            str(instance) + '.dcm')

    pat_name = RefDs.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    print("Patient's name................:", display_name)

    print("Patient id....................:", RefDs.PatientID)

    print("Scan Instance.................:", RefDs.InstanceNumber)

    print("Modality......................:", RefDs.Modality)

    print("BodyPartExamined..............:", RefDs.BodyPartExamined)  

    print("Image Position    (Patient)...:", RefDs.ImagePositionPatient)

    print("Image Orientation (Patient)...:", RefDs.ImageOrientationPatient)

    print("Pixel Spacing.................:", RefDs.PixelSpacing)

    print('Window Center.................:', RefDs.WindowCenter)

    print('Window Width..................:', RefDs.WindowWidth)

    print('Window Intercept..............:', RefDs.RescaleIntercept)



    

interact(explore_patients_metadata, patient_id= patient_ids, instance = (1,150))

# Define a function to visualize the data

def explore_dicoms(patient_id, instance):

    RefDs = pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/' + 

                            patient_id +

                            '/' + 

                            str(instance) + '.dcm')

    plt.figure(figsize=(10, 5))

    

    plt.imshow(RefDs.pixel_array, cmap='gray');

    plt.title(f'P_ID: {patient_id}\nInstance: {instance}')

    plt.axis('off')

interact(explore_dicoms, patient_id= patient_ids, instance = (1,40))