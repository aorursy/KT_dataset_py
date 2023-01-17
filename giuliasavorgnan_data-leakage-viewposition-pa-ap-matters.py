%matplotlib inline
import matplotlib.pyplot as plt
import pylab
import numpy as np
import pydicom
import pandas as pd
from glob import glob
import os

datapath = '../input/'
df_box = pd.read_csv(datapath+'stage_1_train_labels.csv')
print('Number of rows (unique boxes per patient) in main train dataset:', df_box.shape[0])
print('Number of unique patient IDs:', df_box['patientId'].nunique())
df_box.head(6)
df_aux = pd.read_csv(datapath+'stage_1_detailed_class_info.csv')
print('Number of rows in auxiliary dataset:', df_aux.shape[0])
print('Number of unique patient IDs:', df_aux['patientId'].nunique())
df_aux.head(6)
assert df_box['patientId'].values.tolist() == df_aux['patientId'].values.tolist(), 'PatientId columns are different.'
df_train = pd.concat([df_box, df_aux.drop(labels=['patientId'], axis=1)], axis=1)
df_train.head(6)
def get_dcm_data_per_patient(pId):
    '''
    Given one patient ID, 
    return the corresponding dicom data.
    '''
    return pydicom.read_file(datapath+'stage_1_train_images/'+pId+'.dcm')
def get_metadata_per_patient(pId, attribute):
    '''
    Given a patient ID, return useful meta-data from the corresponding dicom image header.
    Return: 
    attribute value
    '''
    # get dicom image
    dcmdata = get_dcm_data_per_patient(pId)
    # extract attribute values
    attribute_value = getattr(dcmdata, attribute)
    return attribute_value
# create list of attributes that we want to extract (manually edited after checking which attributes contained valuable information)
attributes = ['PatientSex', 'PatientAge', 'ViewPosition']
for a in attributes:
    df_train[a] = df_train['patientId'].apply(lambda x: get_metadata_per_patient(x, a))
# convert patient age from string to numeric
df_train['PatientAge'] = df_train['PatientAge'].apply(pd.to_numeric, errors='coerce')
# remove a few outliers
df_train['PatientAge'] = df_train['PatientAge'].apply(lambda x: x if x<120 else np.nan)
df_train.head()
# look at age statistics between positive and negative target groups
df_train.drop_duplicates('patientId').groupby('Target')['PatientAge'].describe()
# look at gender statistics between positive and negative target groups
df_train.drop_duplicates('patientId').groupby(['PatientSex', 'Target']).size() / df_train.drop_duplicates('patientId').groupby(['PatientSex']).size()
# look at patient position statistics between positive and negative target groups
df_train.drop_duplicates('patientId').groupby(['ViewPosition', 'Target']).size() / df_train.drop_duplicates('patientId').groupby(['ViewPosition']).size()
# absolute split of view position
df_train.groupby('ViewPosition').size()