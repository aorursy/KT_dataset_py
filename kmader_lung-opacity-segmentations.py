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
image_bbox_df = pd.merge(comb_box_df, 
                         image_df, 
                         on='patientId',
                        how='left').sort_values('patientId')
print(image_bbox_df.shape[0], 'image bounding boxes')
image_bbox_df.head(5)
from scipy.ndimage import zoom
DCM_TAG_LIST = ['PatientAge', 'BodyPartExamined', 'ViewPosition', 'PatientSex']
def process_dicom(in_path, in_rows):
    c_dicom = pydicom.read_file(in_path, stop_before_pixels=False)
    tag_dict = {c_tag: getattr(c_dicom, c_tag, '') 
         for c_tag in DCM_TAG_LIST}
    tag_dict['path'] = in_path
    tag_dict['PatientAge'] = int(tag_dict['PatientAge'])
    tag_dict['boxes'] = in_rows.shape[0]
    tag_dict['class'] = in_rows['class'].iloc[0]
    return tag_dict, c_dicom.pixel_array
def create_seg_image(base_img, box_rows, out_shape = (512, 512)):
    c_size = base_img.shape
    x_fact = out_shape[0]/c_size[0]
    y_fact = out_shape[1]/c_size[1]
    rs_img = zoom(base_img, (x_fact, y_fact))
    mk_img = np.zeros(rs_img.shape, dtype=bool)
    x_vec = box_rows['x'].map(lambda x: x*x_fact).values.astype(int)
    y_vec = box_rows['y'].map(lambda y: y*y_fact).values.astype(int)
    w_vec = box_rows['width'].map(lambda w: w*x_fact).values.astype(int)
    h_vec = box_rows['height'].map(lambda h: h*y_fact).values.astype(int)
    for x, y, w, h in zip(x_vec,
                         y_vec, 
                         w_vec, 
                         h_vec):
        mk_img[y:(y+h), x:(x+w)] = True
    return rs_img, mk_img
t_path, t_rows = next(iter(image_bbox_df.query('boxes>3').groupby(['path'])))
d_info, d_img = process_dicom(t_path, t_rows)
print(d_info)
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (10, 5))

ax0.imshow(d_img, cmap='bone')
ax0.set_title('Standard Overlay')
for _, c_row in t_rows.iterrows():
    ax0.add_patch(Rectangle(xy=(c_row['x'], c_row['y']),
                 width=c_row['width'],
                 height=c_row['height'],
                           alpha=5e-1))

t_img, t_mask = create_seg_image(d_img, t_rows)
ax1.imshow(t_img)
ax1.set_title("Downscaled Image")
ax2.imshow(t_mask)
ax2.set_title('Mask Image')
t_rows
out_img_size = (512, 512)
keep_patients = 15000
balanced_patient_list = image_bbox_df.\
    query('Target==1')[['Target', 'path']].\
    drop_duplicates().\
    groupby('Target').\
    apply(lambda x: x.sample(keep_patients, random_state=2018, replace=True)).\
    reset_index(drop=True)['path'].values
np.random.seed(2018)
all_groups = list(image_bbox_df[image_bbox_df['path'].isin(balanced_patient_list)].groupby(['path']))
keep_idx = np.random.choice(range(len(all_groups)), keep_patients)
all_groups = [all_groups[i] for i in keep_idx]
import h5py
from tqdm import tqdm_notebook

with h5py.File('train_segs.h5', 'w') as f:
    image_out = f.create_dataset('image', 
                                 shape=(len(all_groups), out_img_size[0], out_img_size[1], 1),
                                 dtype=np.uint8)
    mask_out = f.create_dataset('mask', 
                                 shape=(len(all_groups), out_img_size[0], out_img_size[1], 1),
                                 dtype=bool,
                                 compression='gzip')
    d_info, _ = process_dicom(t_path, t_rows)
    
    key_ds_out = {}
    for k,v in d_info.items():
        if isinstance(v, str):
            key_ds_out[k] = f.create_dataset(k, 
                                 shape=(len(all_groups),),
                                 dtype='S{}'.format(len(v)+2))
        elif isinstance(v, int):
            key_ds_out[k] = f.create_dataset(k, 
                                 shape=(len(all_groups),),
                                 dtype=int)
        else:
            print('Unsupported key-type {}: {}'.format(type(v), v))
    for i, (c_path, c_rows) in enumerate(tqdm_notebook(all_groups)):
        c_info, c_raw_img = process_dicom(c_path, c_rows)
        c_img, c_mask = create_seg_image(c_raw_img, c_rows, out_shape=out_img_size)
        image_out[i, :, :, 0] = c_img
        mask_out[i, :, :, 0] = c_mask
        for k in key_ds_out.keys():
            if k in c_info:
                if isinstance(c_info[k], str):
                    key_ds_out[k][i] = c_info[k].encode('ascii')
                else:
                    key_ds_out[k][i] = c_info[k]
!ls -lh *.h5
with h5py.File('train_segs.h5', 'r') as f:
    for k in f.keys():
        print(k, f[k].shape, f[k].dtype)
        print(k, f[k][0])
