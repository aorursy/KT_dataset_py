%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import pandas as pd
from glob import glob
import os
from matplotlib.patches import Rectangle
det_class_path = '../input/rsna-pneumonia-detection-challenge/stage_1_detailed_class_info.csv'
bbox_path = '../input/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv'
dicom_dir = '../input/rsna-pneumonia-detection-challenge/stage_1_train_images/'
test_dicom_dir = '../input/rsna-pneumonia-detection-challenge/stage_1_test_images/'
image_df = pd.DataFrame({'path': glob(os.path.join(dicom_dir, '*.dcm'))+
                         glob(os.path.join(test_dicom_dir, '*.dcm'))
                        })
image_df['patientId'] = image_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
image_df['data_split'] = image_df['path'].map(lambda x: x.split('/')[-2].split('_')[-2])
print(image_df.shape[0], 'images found')
image_df.sample(3)
DCM_TAG_LIST = ['PatientAge', 'BodyPartExamined', 'ViewPosition', 'PatientSex']
def get_tags(in_path):
    c_dicom = pydicom.read_file(in_path, stop_before_pixels=True)
    tag_dict = {c_tag: getattr(c_dicom, c_tag, '') 
         for c_tag in DCM_TAG_LIST}
    tag_dict['path'] = in_path
    return pd.Series(tag_dict)
image_meta_df = image_df.apply(lambda x: get_tags(x['path']), 1)
# show the summary
image_meta_df['PatientAge'] = image_meta_df['PatientAge'].map(int)
image_meta_df['PatientAge'].hist()
image_meta_df.drop('path',1).describe(exclude=np.number)
rsna_df = pd.merge(image_meta_df, image_df, on='path')
print('Overlapping meta and name data', rsna_df.shape[0])
rsna_df.drop('path',1).describe(exclude=np.number)
nih_data_df = pd.read_csv('../input/data/Data_Entry_2017.csv')
nih_bbox_df = pd.read_csv('../input/data/BBox_List_2017.csv')
nih_path_map = {os.path.basename(x): x for x in glob('../input/data/*/images/*.png')}
print('Number of images', len(nih_path_map))
print('Number of cases', nih_data_df.shape[0])
print('Number of bounding boxes', nih_bbox_df.shape[0])
nih_data_df.sample(3)
simple_nih_df = nih_data_df.rename(columns={'Patient Age': 'PatientAge', 
                         'View Position': 'ViewPosition',
                         'Patient Gender': 'PatientSex'})[['Image Index', 'PatientAge', 'PatientSex', 'ViewPosition', 'Finding Labels']]
simple_nih_df['path'] = simple_nih_df['Image Index'].map(nih_path_map.get)
simple_nih_df.sample(5)
join_nih_fcn = lambda x: pd.merge(x, simple_nih_df, 
                         on=['PatientAge', 'PatientSex', 'ViewPosition'], 
                         suffixes=['_rsna', '_nih'])
test_merge_df = join_nih_fcn(rsna_df.sample(1))
print(test_merge_df.shape[0], 'matches for a single entry in the RSNA dataset')
from skimage.io import imread
from functools import lru_cache
def read_png(in_path):
    return imread(in_path, as_gray=True)
def read_dicom(in_path):
    return pydicom.read_file(in_path).pixel_array
read_dicom_cached = lru_cache(1)(read_dicom) # we have lots of double counting
test_merge_df['rsna_data'] = test_merge_df['path_rsna'].map(read_dicom_cached)
test_merge_df['nih_data'] = test_merge_df['path_nih'].map(read_png)
mean_rsna = test_merge_df['rsna_data'].map(np.mean)
mean_nih = test_merge_df['nih_data'].map(np.mean)
fig, (ax1) = plt.subplots(1, 1, figsize = (8, 4))
ax1.hist(mean_rsna, np.linspace(0, 255, 30), label='RSNA Data')
ax1.hist(mean_nih, np.linspace(0, 255, 30), label='NIH Data', alpha = 0.5)
ax1.legend();
norm_func = lambda x: (x-1.0*np.mean(x)).astype(np.float32)/np.std(x)
mean_rsna = test_merge_df['rsna_data'].map(lambda x: np.max(norm_func(x)))
mean_nih = test_merge_df['nih_data'].map(lambda x: np.max(norm_func(x)))
fig, (ax1) = plt.subplots(1, 1, figsize = (8, 4))
ax1.hist(mean_rsna, np.linspace(0, 3, 30), label='RSNA Data')
ax1.hist(mean_nih, np.linspace(0, 3, 30), label='NIH Data', alpha = 0.5)
ax1.legend();
test_merge_df['img_diff'] = test_merge_df.apply(lambda c_row: np.mean(np.abs(norm_func(c_row['rsna_data'])-norm_func(c_row['nih_data']))),1)
test_merge_df['img_corr'] = test_merge_df.apply(lambda c_row: np.corrcoef(c_row['rsna_data'].ravel(), 
                                                                          c_row['nih_data'].ravel())[0, 1],1)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (21, 7))
ax1.hist(test_merge_df['img_diff'])
ax1.set_title('MAE')
ax2.hist(test_merge_df['img_corr'])
ax2.set_title('Correlation')
ax3.plot(test_merge_df['img_diff'], test_merge_df['img_corr'], '.')
ax3.set_title('MAE vs Correlation')
fig, m_axs = plt.subplots(2, 3, figsize = (25, 10))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for (i, (ax1, ax2)), (_, c_row) in zip(enumerate(m_axs.T, 1),
                                  test_merge_df.sort_values('img_corr', ascending=False).iterrows()
                                 ):
    ax1.imshow(c_row['rsna_data'], cmap='gray')
    ax1.set_title('RSNA Data')
    ax2.imshow(c_row['nih_data'], cmap='gray')
    ax2.set_title('NIH Data Rank:{}\nCorrelation: {:2.1f}%, MAE: {:2.2f}'.format(i, 100*c_row['img_corr'], c_row['img_diff']))
full_merge_df = join_nih_fcn(rsna_df)
print(full_merge_df.shape[0], 'total number of overlapping entires')
matched_groups = rsna_df.groupby(['PatientAge', 'PatientSex', 'ViewPosition'])
print('Number of unique patient age, sex and view combinations:', len(matched_groups))
DS_FACTOR = 4
@lru_cache(maxsize=None)
def read_png_cached(in_path):
    return imread(in_path, as_gray=True)[::DS_FACTOR, ::DS_FACTOR]
@lru_cache(maxsize=None)
def read_dicom_cached(in_path):
    return pydicom.read_file(in_path).pixel_array[::DS_FACTOR, ::DS_FACTOR]
out_matches_list = []
from tqdm import tqdm_notebook
import gc
gc.enable()
gc.collect()
def calc_match_dist(in_row):
    nih_img = read_png_cached(in_row['path_nih'])
    rsna_img = read_dicom_cached(in_row['path_rsna'])
    return np.corrcoef(nih_img.ravel(), rsna_img.ravel())[0, 1]

for _, rsna_grp_rows_df in tqdm_notebook(matched_groups):
    # reset the caches
    gc.collect()
    gr_combo_df = join_nih_fcn(rsna_grp_rows_df)
    gr_combo_df['img_corr'] = gr_combo_df.apply(calc_match_dist,1)
    # add the top match for each group to the list
    out_matches_list += [gr_combo_df.groupby('patientId').\
                             apply(lambda x: x.sort_values('img_corr', ascending=False).head(1)).\
                             reset_index(drop=True)]
    read_dicom_cached.cache_clear()
    read_png_cached.cache_clear()
matches_df = pd.concat(out_matches_list)
matches_df.to_csv('matched_images.csv', index=False)
matches_df.sample(3)
matches_df['img_corr'].plot.hist()
matches_df['Finding Labels'].value_counts()
