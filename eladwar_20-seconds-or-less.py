import numpy as np

import pandas as pd 



import os

import sys



import glob



import cv2

import matplotlib.pyplot as plt



import tensorflow as tf 

import tensorflow.keras.layers as L

from tensorflow.keras.models import Model

import tensorflow.keras.backend as K



from joblib import Parallel, delayed

from tqdm import tqdm_notebook





import vtk

from vtk.util import numpy_support

import numpy
reader = vtk.vtkDICOMImageReader()

def get_img(path):

    reader.SetFileName(path)

    reader.Update()

    _extent = reader.GetDataExtent()

    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]



    ConstPixelSpacing = reader.GetPixelSpacing()

    imageData = reader.GetOutput()

    pointData = imageData.GetPointData()

    arrayData = pointData.GetArray(0)

    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)

    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')

    ArrayDicom = cv2.resize(ArrayDicom,(512,512))

    return ArrayDicom
#store unique patient subdirectories

tr_patient_paths = sorted(glob.glob('/kaggle/input/osic-pulmonary-fibrosis-progression/train/*'))

#Read and Sort : train csv

train = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')

train.sort_values(by = 'Patient',inplace=True)
#Retrieve and list paths in sequential order : ie. (dicom1,dicom2 ...)

all_paths = []

for patient_path in tr_patient_paths:

    organized_paths = glob.glob(patient_path+'/*')

    organized_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    all_paths.extend(organized_paths)
#Make a path DataFrame

image_df = pd.DataFrame(all_paths,columns=['Paths'])

# Retrieve Patient IDs

image_df['Patient'] = image_df['Paths'].apply(lambda x: x.split('/')[5])

# Retrieve Patient Scan Number : eg. What visit they are getting the scan on

image_df['Visit'] = image_df['Paths'].apply(lambda x: int(x.split('/')[-1].split('.')[0]))
#Find the time passed

train['min'] = train.groupby('Patient')['Weeks'].transform('min')

train['time_passed'] = (train['Weeks'] - train['min'])+1

#Find the total treatment period

train['Total_Time_of_Treatment'] = train.groupby('Patient')['time_passed'].transform('max')
max_visits_tr = dict([*zip(image_df.groupby('Patient')['Visit'].max().index.values.tolist(),image_df.groupby('Patient')['Visit'].max().values.tolist())])

#Map each patient to their total amount of visits

train['Max_Visits'] = train['Patient'].map(max_visits_tr)

#Scale each patient visit in the train csv to a particular scan, such as (5 or 136), like in (dicom5 or dicom 136)

train['Visit'] = np.around(train['Max_Visits']*(train['time_passed']/train['Total_Time_of_Treatment']),0).astype(int)
#Drop Patient column so the join works better

image_df.drop('Patient',axis=1,inplace=True)
#Join on the scan from the visit closest to their last measurement --

#And retrieve the paths corresponding to those scans  

tr_im_paths = train.join(image_df,lsuffix='Visit', rsuffix='Visit',how='left')['Paths']
#Check Shape of Scan Paths

tr_im_paths.shape
plt.imshow(get_img(tr_im_paths[1500]))

plt.show()
max_len = len(tr_im_paths)
%%time

X = np.array([Parallel(n_jobs=1)(delayed(get_img)(filename) for filename in tqdm_notebook(tr_im_paths[:max_len]))])[0]

# X = np.array([Parallel(n_jobs=4)(delayed(get_img)(filename) for filename in tqdm_notebook(tr_im_paths[:max_len]))])[0]
y  = train['FVC'][:len(X)].astype(np.float32)
print('Shapes are:',X.shape,y.shape)

print("Memory taken by X and Y in bytes are:",sys.getsizeof(X),sys.getsizeof(y))
18620/128 