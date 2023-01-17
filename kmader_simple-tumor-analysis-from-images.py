import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from skimage.util.montage import montage2d

from skimage.color import label2rgb

from collections import namedtuple

import seaborn as sns

import matplotlib

matplotlib.style.use('ggplot')

import os

import h5py
study_df = pd.read_csv(os.path.join('..', 'input', 'study_list.csv'))

study_df.sample(3) # show 3 random patients
pdata = namedtuple('PatientData', ['ct', 'pet', 'label', 'valid'])

def get_data(patient_id):

    with h5py.File(os.path.join('..', 'input', 'lab_petct_vox_5.00mm.h5'), 'r') as p_data:

        try:

            ct_data = p_data['ct_data'][patient_id]

            pet_data = p_data['pet_data'][patient_id]

            label_data = p_data['label_data'][patient_id]

        except KeyError as ke:

            return pdata(None, None, None, False)

        return pdata(np.array(ct_data), np.array(pet_data), np.array(label_data), True)

vox_size = 5.00 # voxel size (from h5 name) is 5.0mm isotropic

def _ifvalidthen(c_func, elseval = np.NAN):

    def _tfun(patient_id):

        cdata = get_data(patient_id)

        if cdata.valid:

            return c_func(cdata)

        else:

            return elseval

    return _tfun

# calculate the volume by adding up all nonzero voxels in the mask

get_tumor_volume=_ifvalidthen(lambda cdata: np.sum(cdata.label>0)*(vox_size**3))

# calculate the mean PET value inside the tumor

get_mean_tumor_pet=_ifvalidthen(lambda cdata: np.mean(cdata.pet[cdata.label>0]))  

assert get_data('STS_002').valid, "Dataset should be valid"

assert int(get_tumor_volume('STS_002'))==60500, "Tumor size incorrect"

assert int(get_mean_tumor_pet('STS_002')*1000)==4664, "PET value incorrect"
study_df['Tumor Volume (mm3)'] = study_df['Patient ID'].map(get_tumor_volume)

study_df['Tumor Volume (cm3)'] = study_df['Tumor Volume (mm3)']/1000

study_df['Tumor PET (SUV)'] = study_df['Patient ID'].map(get_mean_tumor_pet)

study_df[['Grade', 'Treatment', 'Tumor Volume (cm3)', 'Tumor PET (SUV)']].dropna(0).sample(3)
%matplotlib inline

study_df.plot.scatter(x = 'Age', y = 'Tumor Volume (cm3)')
study_df['Tumor Volume (cm3)'].hist(by = study_df['Grade'], sharex = True)
study_df['Tumor PET (SUV)'].hist(by = study_df['Grade'], sharex = True)