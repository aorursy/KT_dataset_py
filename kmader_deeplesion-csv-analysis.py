%matplotlib inline
from glob import glob
import os, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import seaborn as sns
from collections import defaultdict
base_img_dir = '../input/minideeplesion/'
patient_df = pd.read_csv('../input/DL_info.csv')
patient_df['kaggle_path'] = patient_df.apply(lambda c_row: os.path.join(base_img_dir, 
                                                                        '{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(**c_row),
                                                                        '{Key_slice_index:03d}.png'.format(**c_row)), 1)

print('Loaded', patient_df.shape[0], 'cases')
# extact the bounding boxes
patient_df['bbox'] = patient_df['Bounding_boxes'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1, 4)))
patient_df['norm_loc'] = patient_df['Normalized_lesion_location'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1)))
patient_df['Spacing_mm_px_'] = patient_df['Spacing_mm_px_'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1)))
patient_df['Lesion_diameters_Pixel_'] = patient_df['Lesion_diameters_Pixel_'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1)))
patient_df['Radius_x'] = patient_df.apply(lambda x: x['Lesion_diameters_Pixel_'][0]*x['Spacing_mm_px_'][0], 1)

lesion_type_dict = dict(enumerate('Bone,Abdomen,Mediastinum,Liver,Lung,Kidney,Soft tissue,Pelvis'.split(','), 1))

patient_df['Coarse_lesion_name'] = patient_df['Coarse_lesion_type'].map(lambda x: lesion_type_dict.get(x, 'Unknown'))
for i, ax in enumerate('xyz'):
    patient_df[f'{ax}_loc'] = patient_df['norm_loc'].map(lambda x: x[i])
print('Found', patient_df.shape[0], 'patients with images')
patient_df.sample(3)
sns.pairplot(hue='Coarse_lesion_name', data=patient_df[['Patient_age', 'Coarse_lesion_name', 'Key_slice_index', 'Radius_x']])
freq_flyers_df = patient_df.groupby('Patient_index')[['Patient_age']].apply(
    lambda x: pd.Series({'counts': x.shape[0], 
                         'Age_Range': np.max(x['Patient_age'])-np.min(x['Patient_age'])})).reset_index().sort_values('Age_Range', ascending = False)
sns.pairplot(freq_flyers_df[['counts', 'Age_Range']])
freq_flyers_df.head(5)
join_df = pd.merge(patient_df, freq_flyers_df.head(15))
ax = sns.lmplot(x='Patient_age', y='Radius_x', ci=50,
                hue = 'Coarse_lesion_name',
                sharex=False, sharey=False, x_jitter=0.5,
                 col='Patient_index', col_wrap=5,
                data = join_df)
sns.lmplot(x='Study_index', y='Radius_x', ci=50,
                hue = 'Coarse_lesion_name',
                sharex=False, sharey=False, x_jitter=0.5,
                 col='Patient_index', col_wrap=5,
                data = join_df)
def count_and_check_studies(in_patient_df):
    gr_df = in_patient_df.groupby('Study_index').size().reset_index(name='counts')
    match_df = gr_df[gr_df['counts']==gr_df['counts'].max()]
    
    if (gr_df['counts'].max()>1) and (match_df.shape[0]>1): # more than one study and more than one series 
        return in_patient_df[in_patient_df['Study_index'].isin(match_df['Study_index'])]
    else:
        return in_patient_df.head(0)
grp_patient_df = patient_df.groupby(['Patient_index']).apply(count_and_check_studies).reset_index(drop = True)
print(grp_patient_df.shape[0], 'scans available')
print(len(grp_patient_df['Patient_index'].value_counts()), 'patients')
print(len(grp_patient_df.groupby(['Patient_index', 'Study_index'])), 'studies')
grp_patient_df.head(5)
ff_grp_df = grp_patient_df.groupby(['Patient_index']).size().reset_index(name='counts').sort_values('counts', ascending=False)
join_df = pd.merge(grp_patient_df, ff_grp_df.head(15))
sns.lmplot(x='Study_index', y='Radius_x', ci=50,
                hue = 'Coarse_lesion_name',
                sharex=False, sharey=False,
                 col='Patient_index', col_wrap=5,
                data = join_df)
