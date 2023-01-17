%matplotlib inline
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.segmentation import mark_boundaries
DATA_DIR = os.path.join('..', 'input')
class_str = """car, 33
motorbicycle, 34
bicycle, 35
person, 36
rider, 37
truck, 38
bus, 39
tricycle, 40
others, 0
rover, 1
sky, 17
car_groups, 161
motorbicycle_group, 162
bicycle_group, 163
person_group, 164
rider_group, 165
truck_group, 166
bus_group, 167
tricycle_group, 168
road, 49
siderwalk, 50
traffic_cone, 65
road_pile, 66
fence, 67
traffic_light, 81
pole, 82
traffic_sign, 83
wall, 84
dustbin, 85
billboard, 86
building, 97
bridge, 98
tunnel, 99
overpass, 100
vegatation, 113
unlabeled, 255"""
class_dict = {v.split(', ')[0]: int(v.split(', ')[-1]) for v in class_str.split('\n')}
def get_label_info(in_path):
    idx_image = imread(in_path)//1000
    out_dict = {'dim': idx_image.shape}
    count_dict = {k: np.sum(idx_image==k) for k in np.unique(idx_image)}
    for k,v in class_dict.items():
        out_dict[k] = count_dict.get(v, 0)*1.0/np.prod(idx_image.shape[0:2])
    return out_dict
all_paths = pd.DataFrame(dict(path = glob(os.path.join(DATA_DIR, '*', '*.*p*g'))))
classdict = {0:'others', 1:'rover', 17:'sky', 33:'car', 34:'motorbicycle', 35:'bicycle', 36:'person', 37:'rider', 38:'truck', 39:'bus', 40:'tricycle', 49:'road', 50:'siderwalk', 65:'traffic_cone'}
all_paths['split'] = all_paths['path'].map(lambda x: x.split('/')[-2].split('_')[0])
all_paths['group'] = all_paths['path'].map(lambda x: x.split('/')[-2].split('_')[-1])
all_paths['group'] = all_paths['group'].map(lambda x: 'color' if x == 'test' else x)
all_paths['id'] = all_paths['path'].map(lambda x: '_'.join(os.path.splitext(os.path.basename(x))[0].split('_')[:4]))
all_paths.sample(5)
group_df = all_paths.pivot_table(values = 'path', columns = 'group', aggfunc = 'first', index = ['id', 'split']).reset_index()
group_df.sample(5)
train_df = group_df.query('split=="train"')
print(train_df.shape[0], 'rows')
sample_rows = 6
fig, m_axs = plt.subplots(sample_rows, 3, figsize = (20, 6*sample_rows))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
out_rows = []
for (ax1, ax2, ax3), (_, c_row) in zip(m_axs, train_df.sample(sample_rows).iterrows()):
    c_img = imread(c_row['color'])
    l_img = imread(c_row['label'])//1000
    ax1.imshow(c_img)
    ax1.set_title('Color')
    ax2.imshow(l_img, cmap = 'nipy_spectral')
    ax2.set_title('{car}'.format(**get_label_info(c_row['label'])))
    xd, yd = np.where(l_img>0)
    bound_img = mark_boundaries(image = c_img, label_img = l_img, color = (1,0,0), background_label = 255, mode = 'thick')
    ax3.imshow(bound_img[xd.min():xd.max(), yd.min():yd.max(),:])
    ax3.set_title('Cropped Overlay')
    out_rows += [get_label_info(c_row['label'])]
pd.DataFrame(out_rows)
def read_row(in_row):
    return dict(**in_row, **get_label_info(in_row['label']))
%%time
all_rows = []
for _, c_row in list(train_df.sample(40).iterrows()):
    all_rows += [read_row(c_row.to_dict())]
%%time
from dask import bag
some_rows = bag.from_sequence([x.to_dict() for _, x in train_df.sample(40).iterrows()]).map(read_row)
_ = some_rows.compute()
all_rows = bag.from_sequence([x.to_dict() for _, x in train_df.iterrows()], npartitions = 10000).map(read_row)
%%time
all_rows_df = pd.DataFrame(all_rows.compute())
ordered_cols = list(train_df.columns)+['dim']
for c_col in all_rows_df.columns:
    if c_col not in ordered_cols:
        ordered_cols += [c_col]
all_rows_df = all_rows_df[ordered_cols]
all_rows_df.to_csv('label_breakdown.csv')
all_rows_df.sample(5)
import seaborn as sns
all_keys = list(class_dict.keys())
for i in range(0, len(all_keys), 4):
    sns.pairplot(all_rows_df[all_keys[i:(i+4)]])
