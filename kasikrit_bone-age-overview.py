import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob
# not needed in Kaggle, but required in Jupyter
%matplotlib inline 
base_bone_dir = os.path.join('..', 'input')
age_df = pd.read_csv(os.path.join(base_bone_dir, 'boneage-training-dataset.csv'))
age_df['path'] = age_df['id'].map(lambda x: os.path.join(base_bone_dir,
                                                         'boneage-training-dataset', 
                                                         'boneage-training-dataset', 
                                                         '{}.png'.format(x)))
age_df['exists'] = age_df['path'].map(os.path.exists)
print(age_df['exists'].sum(), 'images found of', age_df.shape[0], 'total')
age_df['gender'] = age_df['male'].map(lambda x: 'male' if x else 'female')
age_df.dropna(inplace = True)
age_df.sample(3)
age_df[['boneage', 'gender']].hist(figsize = (10, 5))
age_groups = 8
age_df['age_class'] = pd.qcut(age_df['boneage'], age_groups)
age_overview_df = age_df.groupby(['age_class', 
                                  'gender']).apply(lambda x: x.sample(1)
                                                             ).reset_index(drop = True
                                                                          )

age_overview_df
fig, m_axs = plt.subplots( age_groups, 2, figsize = (12, 6*age_groups))
for c_ax, (_, c_row) in zip(m_axs.flatten(), 
                            age_overview_df.sort_values(['age_class', 'gender']).iterrows()):
    c_ax.imshow(imread(c_row['path']),
                cmap = 'viridis')
    c_ax.axis('off')
    c_ax.set_title('{boneage} months, {gender}'.format(**c_row))
fig, m_axs = plt.subplots( age_groups, 2, figsize = (12, 6*age_groups))
for c_ax, (_, c_row) in zip(m_axs.flatten(), 
                            age_overview_df.sort_values(['age_class', 'gender']).iterrows()):
    c_ax.imshow(imread(c_row['path']),
                cmap = 'bone')
    c_ax.axis('off')
    c_ax.set_title('{boneage} months, {gender}'.format(**c_row))
fig, m_axs = plt.subplots( age_groups, 2, figsize = (12, 6*age_groups))
for c_ax, (_, c_row) in zip(m_axs.flatten(), 
                            age_overview_df.sort_values(['age_class', 'gender']).iterrows()):
    c_ax.imshow(imread(c_row['path']),
                cmap = 'gray')
    c_ax.axis('off')
    c_ax.set_title('{boneage} months, {gender}'.format(**c_row))
