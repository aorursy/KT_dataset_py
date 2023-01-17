# Import Packages
!pip install fastai2 -q
import os
from fastai2.medical.imaging import *
from PIL import Image
import pydicom
TRAIN_ROOT = '../input/osic-pulmonary-fibrosis-progression/train'
PATIENT_ID = 'ID00010637202177584971671'
dcm_img = dcmread(os.path.join(TRAIN_ROOT,PATIENT_ID,'1.dcm'))
dcm_img
for PATIENT_ID in os.listdir(TRAIN_ROOT)[:3]:
    for dcm_img_path in os.listdir(os.path.join(TRAIN_ROOT,PATIENT_ID)):
        dcm_img = dcmread(os.path.join(TRAIN_ROOT,PATIENT_ID,dcm_img_path))
        if hasattr(dcm_img,'ImageType'):
            orientation_list = list(dcm_img['ImageType'])
            print(orientation_list)
for PATIENT_ID in os.listdir(TRAIN_ROOT):
    for dcm_img_path in os.listdir(os.path.join(TRAIN_ROOT,PATIENT_ID)):
        dcm_img = dcmread(os.path.join(TRAIN_ROOT,PATIENT_ID,dcm_img_path))
        if hasattr(dcm_img,'ImageOrientationPatient'):
            orientation_list = list(dcm_img['ImageOrientationPatient'])
            orientation_list = [int(i) for i in orientation_list]
            if orientation_list != [1, 0, 0, 0, 1, 0]:
                print(PATIENT_ID)
                print(orientation_list)
                break