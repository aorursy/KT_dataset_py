%matplotlib inline
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
label_names = {0: 'unlabeled', 1: 'man-made terrain', 2: 'natural terrain', 
               3: 'high vegetation', 4: 'low vegetation', 5: 'buildings', 
               6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}
DATA_DIR = os.path.join('..', 'input')
test_path = os.path.join(DATA_DIR, 'bildstein_station1_xyz_intensity_rgb.h5')
train_path = os.path.join(DATA_DIR, 'domfountain_station1_xyz_intensity_rgb.h5')

def _fix_name(k):
    if k=='r': return 'red'
    if k=='g': return 'green'
    if k=='b': return 'blue'
    return k
def read_as_df(in_path, only_valid = False):
    with h5py.File(in_path) as h:
        cur_df = pd.DataFrame({_fix_name(k): h[k].value for k in h.keys()})
        cur_df = cur_df[['x', 'y', 'z', 'red', 'green', 'blue', 'intensity', 'class']]
        if only_valid:
            return cur_df[cur_df['class']>0]
        else:
            return cur_df
test_df = read_as_df(test_path)
print('Testing Points', test_df.shape)
test_df.sample(3)
from pyntcloud import PyntCloud
cloud = PyntCloud(test_df)
cloud
tiny_cloud = PyntCloud(test_df.sample(5000))
tiny_cloud.plot(point_size=0.1, opacity=0.6)
voxelgrid_id = cloud.add_structure("voxelgrid", n_x=128, n_y=128, n_z=128)
voxelgrid = cloud.structures[voxelgrid_id]
cloud
voxelgrid.plot(d=3, mode="density", cmap="hsv")
binary_feature_vector = voxelgrid.get_feature_vector(mode="binary")
binary_feature_vector.shape
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 10))
ax1.imshow(np.sum(binary_feature_vector, 0))
ax2.imshow(np.sum(binary_feature_vector, 1))
ax3.imshow(np.sum(binary_feature_vector, 2))
density_feature_vector = voxelgrid.get_feature_vector(mode="density")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 10))
ax1.imshow(np.sum(density_feature_vector, 0))
ax2.imshow(np.sum(density_feature_vector, 1))
ax3.imshow(np.sum(density_feature_vector, 2))
col_mat = []
for c_color, not_color in zip(['red','green', 'blue'], 
                              [('green', 'blue'), ('red','blue'), ('red','green')]):
    cur_df = test_df[(test_df[c_color]>=test_df[not_color[0]]) & (test_df[c_color]>=test_df[not_color[1]])]
    cur_cloud = PyntCloud(cur_df)
    cur_voxelgrid_id = cur_cloud.add_structure("voxelgrid", n_x=128, n_y=128, n_z=128)
    cur_voxelgrid = cur_cloud.structures[cur_voxelgrid_id]
    col_mat += [cur_voxelgrid.get_feature_vector(mode="binary")]
col_mat = np.stack(col_mat, -1)
print(col_mat.shape)
def sum_rgb(in_img, axis, w_fact = 1):
    out_rgb = np.sum(in_img, axis).astype(np.float32)
    for i in range(out_rgb.shape[2]):
        out_rgb[:, :, i] -= np.mean(out_rgb[:, :, i])
        out_rgb[:, :, i] /= (w_fact*np.std(out_rgb[:, :, i]))
        out_rgb[:, :, i] = out_rgb[:, :, i]*127+127
    return np.clip(out_rgb, 0, 255).astype(np.uint8)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 10))
ax1.imshow(sum_rgb(col_mat, 0))
ax2.imshow(sum_rgb(col_mat, 1))
ax3.imshow(sum_rgb(col_mat, 2))
# build the maps ourselves
n_col_mat = []
from pyntcloud.utils.numba import groupby_sum, groupby_count, groupby_max
rev_lookup = {k: i for i,k in enumerate(cloud.points.columns)}
for c_color in ['red','green', 'blue']:
    s = np.zeros(voxelgrid.n_voxels)
    c = np.zeros(voxelgrid.n_voxels)
    n_col_mat += [(np.nan_to_num(groupby_sum(cloud.points.values, voxelgrid.voxel_n, rev_lookup[c_color] , s) /
                                      groupby_count(cloud.points.values, voxelgrid.voxel_n, c))).reshape(voxelgrid.x_y_z)]
n_col_mat = np.stack(n_col_mat, -1)
print(n_col_mat.shape)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 10))
w_fact = 6
ax1.imshow(sum_rgb(n_col_mat, 0, w_fact=w_fact))
ax2.imshow(sum_rgb(n_col_mat, 1, w_fact=w_fact))
ax3.imshow(sum_rgb(n_col_mat, 2, w_fact=w_fact))
kdtree_id = cloud.add_structure("kdtree")
kdtree = cloud.structures[kdtree_id]
try:
    qhull_id = cloud.add_structure("convexhull")
    qhull = cloud.structures[qhull_id]
except Exception as e:
    print(e)
from pyntcloud.structures import delanuay
#delanuay.Delaunay(tiny_cloud.points) # not working yet

