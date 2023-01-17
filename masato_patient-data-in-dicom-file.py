import numpy as np
import pandas as pd
import sys, time, os, json

import matplotlib.pyplot as plt
from  tqdm import tqdm

import pydicom


class Configs():
    def __init__(self):
        self.data_root_dir = '../input/'
        self.test_image_dir = os.path.join(self.data_root_dir, 'stage_1_test_images/')
        self.train_image_dir = os.path.join(self.data_root_dir, 'stage_1_train_images/')
        self.file_ext = '.dcm'

C = Configs()

print('\n'.join(os.listdir("../input")))
labels = pd.read_csv(C.data_root_dir+'stage_1_train_labels.csv', 
                     dtype={'patientId':str, 
                            'x':np.float, 
                            'y': np.float, 
                            'width': np.float, 
                            'height': np.float, 
                            'target': np.int})

details = pd.read_csv(C.data_root_dir+'stage_1_detailed_class_info.csv', dtype={'patientId': str, 'class': str})
# rename column name to avoid a trouble.
# column name 'class' could cause the trouble when use query()
details.columns = ['patientId', 'details']
whole_label_info = pd.concat([labels, details.drop('patientId', axis=1)], axis=1)

# quick-check
whole_label_info['details'].value_counts()
patientId = whole_label_info['patientId'].drop_duplicates().values
patient_info = pd.DataFrame()
for p in tqdm(patientId):
    ds = pydicom.dcmread(C.train_image_dir + p + C.file_ext)
    tmp_info = pd.DataFrame({
        'patientId': [ds.PatientID],
        'age': [ds.PatientAge],
        'sex': [ds.PatientSex]
    })
    patient_info = patient_info.append(tmp_info)


whole_label_info = whole_label_info.merge(patient_info, on='patientId', how='left')
stat_age = patient_info['age'].value_counts(dropna=False).reset_index().rename({'index': 'age', 'age': 'count'}, axis=1)
stat_age['age'] = stat_age['age'].astype(int)
stat_age.sort_values('age')

print('range of age: ', stat_age['age'].min(), stat_age['age'].max())

plt.bar(stat_age['age'], stat_age['count'])
plt.ylabel('count')
plt.xlabel('age')
plt.show()
stat_sex = patient_info['sex'].value_counts(dropna=False).reset_index().rename({'index': 'sex', 'sex': 'count'}, axis=1)
plt.bar(stat_sex['sex'], stat_sex['count'])
plt.ylabel('count')
plt.xlabel('gender')
plt.show()