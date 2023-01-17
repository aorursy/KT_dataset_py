# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import cm

from matplotlib import pyplot as plt

%matplotlib inline

import cv2

import seaborn as sns

from tqdm import tqdm

from pprint import pprint



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import glob

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pydicom

from pydicom.filereader import read_dicomdir

from pydicom.data import get_testdata_files

import os

from os.path import dirname, join
import zipfile
Dataset = "Neurohacking_data-0.0"



# Unzip the files and place them in the working directory

with zipfile.ZipFile("../input/neurohackinginrimages/"+Dataset+".zip","r") as z:

    z.extractall(".")
from subprocess import check_output

print(check_output(["ls", "Neurohacking_data-0.0"]).decode("utf8"))
print(check_output(["more", "Neurohacking_data-0.0/README.csv"]).decode("utf8"))
print(check_output(["ls", '-l', "Neurohacking_data-0.0/BRAINIX/"]).decode("utf8"))
print(check_output(["ls", '-l', "Neurohacking_data-0.0/BRAINIX/DICOM"]).decode("utf8"))
print(check_output(["ls", '-l', "Neurohacking_data-0.0/BRAINIX/NIfTI"]).decode("utf8"))
print(check_output(["ls", "Neurohacking_data-0.0/BRAINIX/DICOM/T1"]).decode("utf8"))
print(check_output(["ls", "Neurohacking_data-0.0/BRAINIX/DICOM/T2"]).decode("utf8"))
print(check_output(["ls", "Neurohacking_data-0.0/BRAINIX/DICOM/FLAIR"]).decode("utf8"))
print(check_output(["ls", '-l', "Neurohacking_data-0.0/kirby21"]).decode("utf8"))
print(check_output(["ls", '-l', "Neurohacking_data-0.0/kirby21/visit_2/113"]).decode("utf8"))
print(check_output(["ls", 'Neurohacking_data-0.0/BRAINIX/DICOM/T1']))
list_dir = os.listdir('Neurohacking_data-0.0/BRAINIX/DICOM/T1/')

print('len ', list_dir)
def show_dcm_info(dataset):

    #print("Filename.........:", file_path)

    print("Storage type.....:", dataset.SOPClassUID)

    print()



    pat_name = dataset.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    print("Patient's name......:", display_name)

    print("Patient id..........:", dataset.PatientID)

    #print("Patient's Age.......:", dataset.PatientAge)

    print("Patient's Sex.......:", dataset.PatientSex)

    print("Modality............:", dataset.Modality)

    #print("Body Part Examined..:", dataset.BodyPartExamined)

    #print("View Position.......:", dataset.ViewPosition)

    #print("Rescale intercept .......:", dataset.RescaleIntercept)

    #print("Rescale slope .......:", dataset.RescaleSlope)

    

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing mm ....:", dataset.PixelSpacing)

        if 'SliceThickness' in dataset:

            print("Slice thickness mm....:", dataset.SliceThickness)
def plot_pixel_array(dataset, figsize=(10,10)):

    plt.figure(figsize=figsize)

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.show()



def plot_threshold_array(dataset, figsize=(10,10)):

    plt.figure(figsize=figsize)

    plt.imshow(dataset)   #, cmap=plt.cm.brain)

    plt.show()

    

def plot_T1T2FLAIR_array(datasetT1, datasetT2, datasetFLAIR, figsize=(35,25)):

    plt.figure(figsize=figsize)



    #subplot(r,c) provide the no. of rows and columns

    #f, axarr = plt.subplots(1,3) 

    fig, axs = plt.subplots(1, 3, figsize=(40, 40), constrained_layout=True)



    # use the created array to output your multiple images. 

    axs[0].imshow(datasetT1)

    axs[0].set_title('T1 weighted sequence', fontsize=42)

    axs[1].imshow(datasetT2)

    axs[1].set_title('T2 weighted sequence', fontsize=42)

    axs[2].imshow(datasetFLAIR)

    axs[2].set_title('FLAIR sequence', fontsize=42)



    #plt.imshow(dataset)   #, cmap=plt.cm.brain)

    plt.show()
list_dir_T1 = glob.glob("Neurohacking_data-0.0/BRAINIX/DICOM/T1/*.dcm")

list_dir_T2 = glob.glob("Neurohacking_data-0.0/BRAINIX/DICOM/T2/*.dcm")

list_dir_FLAIR = glob.glob("Neurohacking_data-0.0/BRAINIX/DICOM/FLAIR/*.dcm")

sorted_listT1    = list_dir_T1.sort()

sorted_listT2    = list_dir_T2.sort()

sorted_listFLAIR = list_dir_FLAIR.sort()

for file_nameT1, file_nameT2, file_nameFLAIR, in zip(list_dir_T1, list_dir_T2, list_dir_FLAIR):

    print('Reading T1 T2 FLAIR:  ', file_nameT1, file_nameT2, file_nameFLAIR)

    datasetT1 = pydicom.read_file(file_nameT1)

    datasetT2 = pydicom.read_file(file_nameT2)

    datasetFLAIR = pydicom.read_file(file_nameFLAIR)

    

    show_dcm_info(datasetT1)

    plot_T1T2FLAIR_array(datasetT1.pixel_array, datasetT2.pixel_array, datasetFLAIR.pixel_array)