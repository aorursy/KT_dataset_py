%matplotlib inline
import numpy as np # linear algebra
np.random.seed(2018)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob 
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from skimage.util.montage import montage2d
from skimage.io import imread
from tqdm import tqdm
base_dir = os.path.join('..', 'input') # 'pulmonary-chest-xray-abnormalities'
all_xray_df = pd.read_csv(os.path.join(base_dir, 'Data_Entry_2017.csv'))

all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('..', 'input',  'images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

with open(os.path.join(base_dir, 'test_list.txt'), 'r') as f:
    validated_files = [x.strip() for x in f.readlines()]
all_xray_df['validated'] = all_xray_df['Image Index'].isin(validated_files)
all_xray_df['Single Finding'] = all_xray_df['Finding Labels'].map(lambda x: '|' not in x)
print('Single Finding Scans:', all_xray_df['Single Finding'].sum())
all_xray_df.sample(5)
all_xray_df[all_xray_df['Single Finding']].groupby(['Finding Labels'])\
    .size().reset_index(name='counts').sort_values('counts', ascending=False)
unnecessary_categories = ['Mass', 'Cardiomegaly', 'Hernia', 
                          'Nodule', 'Fibrosis']
keep_categories = ['Cardiomegaly', 'Pneumothorax', 'Infiltration', 'Effusion', 'No Finding']
sf_validated_df = all_xray_df[
    all_xray_df['Single Finding'] & 
    all_xray_df['validated'] &
    all_xray_df['Finding Labels'].map(lambda x: x in keep_categories)
]
sf_validated_df.sample(3)
import zipfile as zf
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm_notebook
all_rows = []
DS_SIZE = 512
GROUP_SIZE = 20
AGE_SPLITS = 5
with zf.ZipFile('high_res.zip', 'w') as hrz, zf.ZipFile('low_res.zip', 'w') as lrz:
    for finding, rows_df in tqdm_notebook(sf_validated_df.groupby('Finding Labels')):
        # only 1 per patient
        clean_rows_df = rows_df.groupby('Patient ID').apply(lambda x: x.sample(1)).reset_index(drop=True)
        # group by age and gender
        clean_rows_df['age_group'] = pd.qcut(
            clean_rows_df['Patient Age'], AGE_SPLITS)
        ag_groups = clean_rows_df.groupby(['age_group', 
                                           'Patient Gender'])
        
        if len(ag_groups)<GROUP_SIZE:
            # for some patients the count is very low
            # and so we cant really stratify
            out_rows = clean_rows_df.sample(GROUP_SIZE)
        else:
            # if we have enough stratify by age and gender
            out_rows = ag_groups.apply(lambda x: 
                                       x.sample(GROUP_SIZE//AGE_SPLITS)).reset_index(drop=True)
        
        print(finding, out_rows.shape[0],'/',clean_rows_df.shape[0], 'cases')
        arc_path = lambda x: os.path.basename(x)
        for _, c_row in out_rows.iterrows():
            hrz.write(c_row['path'],
                arcname = arc_path(c_row['path']),
                compress_type = zf.ZIP_STORED)
            
            full_image = imread(c_row['path'], as_grey=True)
            rs_img = resize(full_image, (DS_SIZE, DS_SIZE))
            rgb_rs = plt.cm.gray(rs_img)[:, :, :3]
            imsave('test.png', rgb_rs)
            lrz.write('test.png',
                arcname = arc_path(c_row['path']),
                compress_type = zf.ZIP_STORED)
        all_rows += [out_rows]
        
dataset_df = pd.concat(all_rows)
dataset_df.to_csv('dataset_overview.csv', index=False)
dataset_df.sample(3)
dataset_df.groupby('Finding Labels').size().reset_index(name='counts')
','.join(dataset_df.groupby('Finding Labels').size().reset_index(name='counts')['Finding Labels'].values)
sns.swarmplot(y='Finding Labels', 
               x = 'Patient Age', 
               hue = 'Patient Gender',
               data = dataset_df)
sns.swarmplot(y='Finding Labels', 
               x = 'Patient Age', 
               hue = 'View Position',
               data = dataset_df)
