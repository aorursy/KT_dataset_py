import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
PATH="../input/"
print(os.listdir(PATH))
study_df = pd.read_csv(os.path.join(PATH, 'study_list.csv'))
print("Study list: %2d rows, %2d columns" % (study_df.shape[0], study_df.shape[1]))
study_df
maxImgSet = 0
with h5py.File(os.path.join(PATH, 'patient_images_lowres.h5'), 'r') as patient_data:
    for (patient_id, patient_img) in patient_data['ct_data'].items():
        maxImgSet = max(maxImgSet, len(patient_img))
        print("Patient:", patient_id, " Images:", len(patient_img))
print("\nLargest number of images:",maxImgSet)      
%matplotlib inline
with h5py.File(os.path.join(PATH, 'patient_images_lowres.h5'), 'r') as patient_data:
    for (patient_id, patient_img) in patient_data['ct_data'].items():
        crt_row_df = study_df[study_df['Patient ID']==patient_id]
        print(list(crt_row_df.T.to_dict().values()))
        fig, ax = plt.subplots(13,12, figsize=(13,12), dpi = 250)
        for i, crt_patient_img in enumerate(patient_img):
            ax[i//12, i%12].imshow(crt_patient_img, cmap = 'bone')
            ax[i//12, i%12].axis('off')
        plt.subplots_adjust(hspace = .1, wspace = .1)
        plt.show()
maxImgSet = 0
with h5py.File(os.path.join(PATH, 'lab_petct_vox_5.00mm.h5'), 'r') as patient_data:
    for (patient_id, patient_img) in patient_data['ct_data'].items():
        maxImgSet = max(maxImgSet, len(patient_img))
        print("Patient:", patient_id, " Images:", len(patient_img))
print("\nLargest number of images:",maxImgSet)      
%matplotlib inline
with h5py.File(os.path.join(PATH, 'lab_petct_vox_5.00mm.h5'), 'r') as patient_data:
    for (patient_id, patient_img) in patient_data['ct_data'].items():
        crt_row_df = study_df[study_df['Patient ID']==patient_id]
        print(list(crt_row_df.T.to_dict().values()))
        fig, ax = plt.subplots(17,12, figsize=(13,12), dpi = 250)
        for i, crt_patient_img in enumerate(patient_img):
            ax[i//12, i%12].imshow(crt_patient_img, cmap = 'bone')
            ax[i//12, i%12].axis('off')
        plt.subplots_adjust(hspace = .1, wspace = .1)
        plt.show()