!pip install --upgrade fastai2 > /dev/null
from fastai2.basics import *

from fastai2.callback.all import *

from fastai2.vision.all import *

from fastai2.medical.imaging import *
import pydicom
import numpy as np

import pandas as pd

import os
import cv2
source = Path("../input/siim-isic-melanoma-classification")

files = os.listdir(source)

files
train = source/'train'

train_files = get_dicom_files(train)

train_files
image = train_files[42]
dimg = dcmread(image)
dimg
def show_one_patient(file):

    """ function to view patient image and choosen tags within the head of the DICOM"""

    pat = dcmread(file)

    print(f'patient Name: {pat.PatientName}')

    print(f'Patient ID: {pat.PatientID}')

    print(f'Patient age: {pat.PatientAge}')

    print(f'Patient Sex: {pat.PatientSex}')

    print(f'Body part: {pat.BodyPartExamined}')

    trans = Transform(Resize(256))

    dicom_create = PILDicom.create(file)

    dicom_transform = trans(dicom_create)

    return show_image(dicom_transform)
patient = dcmread(image)
print(f'Photometric Interpretation: {patient.PhotometricInterpretation}')
dicom_create = PILDicom.create(image)
dicom_create.show(figsize=(6,6), cmap=plt.cm.gist_ncar)
pil_image = Image.fromarray(dcmread(image).pixel_array)
open_cv_image = np.array(pil_image) 
open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_YCrCb2BGR)
pil_image = Image.fromarray(open_cv_image)
plt.imshow(pil_image)
def show_one_patient_RGB(file):

    """ function to view patient image and choosen tags within the head of the DICOM"""

    pat = dcmread(file)

    print(f'patient Name: {pat.PatientName}')

    print(f'Patient ID: {pat.PatientID}')

    print(f'Patient age: {pat.PatientAge}')

    print(f'Patient Sex: {pat.PatientSex}')

    print(f'Body part: {pat.BodyPartExamined}')

    trans = Transform(Resize(256))

    

    pil_image = Image.fromarray(dcmread(image).pixel_array)

    # Not sure yet about the use of the following line. For now uncommented.

    # pil_image = trans(pil_image)

    open_cv_image = np.array(pil_image) 

    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_YCrCb2BGR)

    pil_image = Image.fromarray(open_cv_image)

    # Could add a parameter to specify figize. For now it is set to (6,6)

    

    return show_image(pil_image, figsize=(6,6))
show_one_patient_RGB(image)