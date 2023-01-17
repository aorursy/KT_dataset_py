%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import pandas as pd
from glob import glob
import os
from matplotlib.patches import Rectangle
det_class_path = '../input/stage_1_detailed_class_info.csv'
bbox_path = '../input/stage_1_train_labels.csv'
dicom_dir = '../input/stage_1_train_images/'
test_dicom_dir = '../input/stage_1_test_images/'
det_class_df = pd.read_csv(det_class_path)
print(det_class_df.shape[0], 'class infos loaded')
print(det_class_df['patientId'].value_counts().shape[0], 'patient cases')
det_class_df.groupby('class').size().plot.bar()
det_class_df.sample(3)
bbox_df = pd.read_csv(bbox_path)
print(bbox_df.shape[0], 'boxes loaded')
print(bbox_df['patientId'].value_counts().shape[0], 'patient cases')
bbox_df.sample(3)
# we first try a join and see that it doesn't work (we end up with too many boxes)
comb_bbox_df = pd.merge(bbox_df, det_class_df, how='inner', on='patientId')
print(comb_bbox_df.shape[0], 'combined cases')
comb_bbox_df = pd.concat([bbox_df, 
                        det_class_df.drop('patientId',1)], 1)
print(comb_bbox_df.shape[0], 'combined cases')
comb_bbox_df.sample(3)
box_df = comb_bbox_df.groupby('patientId').\
    size().\
    reset_index(name='boxes')
comb_box_df = pd.merge(comb_bbox_df, box_df, on='patientId')
box_df.\
    groupby('boxes').\
    size().\
    reset_index(name='patients')
comb_bbox_df.groupby(['class', 'Target']).size().reset_index(name='Patient Count')
image_df = pd.DataFrame({'path': glob(os.path.join(dicom_dir, '*.dcm'))})
image_df['patientId'] = image_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
print(image_df.shape[0], 'images found')
img_pat_ids = set(image_df['patientId'].values.tolist())
box_pat_ids = set(comb_box_df['patientId'].values.tolist())
# check to make sure there is no funny business
assert img_pat_ids.union(box_pat_ids)==img_pat_ids, "Patient IDs should be the same"
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
test_image_df = pd.DataFrame({'path': glob(os.path.join(test_dicom_dir, '*.dcm'))})
test_image_meta_df = test_image_df.apply(lambda x: get_tags(x['path']), 1)
# show the summary
test_image_meta_df['PatientAge'] = test_image_meta_df['PatientAge'].map(int)
test_image_meta_df['PatientAge'].hist()
test_image_meta_df.to_csv('test_stats.csv')
test_image_meta_df.drop('path',1).describe(exclude=np.number)
image_full_df = pd.merge(image_df,
                         image_meta_df,
                         on='path')
image_bbox_df = pd.merge(comb_box_df, 
                         image_full_df, 
                         on='patientId',
                        how='left')
print(image_bbox_df.shape[0], 'image bounding boxes')
image_bbox_df.sample(3)
image_bbox_df = image_bbox_df[image_bbox_df['class'].isin(['Normal', 'Lung Opacity'])]
sample_df = image_bbox_df.\
    groupby(['Target','class', 'boxes']).\
    apply(lambda x: x[x['patientId']==x.sample(1)['patientId'].values[0]]).\
    reset_index(drop=True)
sample_df
fig, m_axs = plt.subplots(2, 3, figsize = (20, 10))
for c_ax, (c_path, c_rows) in zip(m_axs.flatten(),
                    sample_df.groupby(['path'])):
    c_dicom = pydicom.read_file(c_path)
    c_ax.imshow(c_dicom.pixel_array, cmap='bone')
    c_ax.set_title('{class}'.format(**c_rows.iloc[0,:]))
    for i, (_, c_row) in enumerate(c_rows.dropna().iterrows()):
        c_ax.plot(c_row['x'], c_row['y'], 's', label='{class}'.format(**c_row))
        c_ax.add_patch(Rectangle(xy=(c_row['x'], c_row['y']),
                                width=c_row['width'],
                                height=c_row['height'], 
                                 alpha = 0.5))
        if i==0: c_ax.legend()
import zipfile as zf
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm_notebook
import warnings
warnings.simplefilter('default')

all_rows = []
DS_SIZE = 512
GROUP_SIZE = 25
AGE_SPLITS = 5
with zf.ZipFile('high_res.zip', 'w') as hrz, zf.ZipFile('low_res.zip', 'w') as lrz:
    for finding, rows_df in tqdm_notebook(image_bbox_df.groupby(['class'])):
        # only 1 per patient
        clean_rows_df = rows_df.groupby('patientId').apply(lambda x: x.sample(1)).reset_index(drop=True)
        # group by age and gender
        clean_rows_df['age_group'] = pd.qcut(
            clean_rows_df['PatientAge'], AGE_SPLITS)
        ag_groups = clean_rows_df.groupby(['age_group', 
                                           'PatientSex'])
        
        if len(ag_groups)<GROUP_SIZE:
            # for some patients the count is very low
            # and so we cant really stratify
            out_rows = clean_rows_df.sample(min(GROUP_SIZE, clean_rows_df.shape[0]))
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
            full_image = pydicom.read_file(c_row['path']).pixel_array
            full_image = full_image/full_image.max()
            rs_img = resize(full_image, (DS_SIZE, DS_SIZE))
            rgb_rs = plt.cm.gray(rs_img)[:, :, :3]
            imsave('test.png', rgb_rs)
            lrz.write('test.png',
                arcname = arc_path(c_row['path']),
                compress_type = zf.ZIP_STORED)
        out_rows['Image Index'] = out_rows['path'].map(arc_path)
        all_rows += [out_rows]
dataset_df = pd.concat(all_rows)
dataset_df.to_csv('dataset_overview.csv', index=False)
dataset_df.sample(3)
import seaborn as sns
sns.swarmplot(y='Target', 
               x = 'PatientAge', 
               hue = 'PatientSex',
               data = dataset_df)
dataset_df.groupby('Target').size().reset_index(name='counts')
import json
annotation_task = {
    'google_forms': {'form_url': 'https://docs.google.com/forms/d/e/1FAIpQLSemGag9uitnPnV6OBHDNgrvr2nh-jArJZhVco0Kfjkx4eRkYA/viewform', 
    'sheet_url': 'https://docs.google.com/spreadsheets/d/1JUCLX_17JIGit0Nk4wphgTHlmji9u9PYPmyf_9Wscvg/edit?usp=sharing'
            },
    'dataset': {
        'image_path': 'Image Index', # column name
        'output_labels': 'class', # column name
        'dataframe': dataset_df.drop('age_group', 1).to_dict(),
        'base_image_directory': 'sample_data', # path
        'questions': {'Normal': 'Is there no opacity present in this image?', 
                      'Lung Opacity': 'Is there an opacity present in this image?'} 
    }
}

with open('task.json', 'w') as f:
    json.dump(annotation_task, f, indent=4, sort_keys=True)
