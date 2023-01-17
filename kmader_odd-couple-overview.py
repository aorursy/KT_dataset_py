%matplotlib inline
import os
import numpy as np
from skimage.io import imread
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, ball
from skimage.morphology import label
from skimage.segmentation import mark_boundaries
from skimage.util.montage import montage2d
from IPython.display import display, Markdown
display_md = lambda x: display(Markdown(x))
montage_pad = lambda x: montage2d(np.pad(x, [(0,0), (10, 10), (10, 10)], mode = 'constant', constant_values = 0))
import gc # since memory gets tight very quickly
gc.enable()
base_dir = os.path.join('..', 'input')
all_slices = glob(os.path.join(base_dir, '*', '*', '*'))
print('Number of slices', len(all_slices))
slice_df = pd.DataFrame(dict(path = all_slices))
slice_df['image_type'] = slice_df['path'].map(lambda x: x.split('/')[-3])
slice_df['filename'] = slice_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
slice_df['slice_no'] = slice_df['filename'].map(lambda x: int(x.split('_')[-1]))
print('Sample Rows')
display(slice_df.sample(3))
print('Group Count')
display(slice_df.groupby('image_type').count())
paired_slices_df = slice_df.pivot_table(values = 'path', columns = 'image_type', index = 'slice_no', aggfunc='first').reset_index().dropna()
paired_slices_df.sample(3)
offset_slice_df = slice_df.copy()
offset_slice_df['slice_no'] = offset_slice_df.apply(lambda x: x['slice_no']+(250 if x['image_type']=='binned_image' else 0), 1)
paired_slices_df = offset_slice_df.pivot_table(values = 'path', columns = 'image_type', index = 'slice_no', aggfunc='first').reset_index().dropna().sort_values('slice_no')
test_row = next(paired_slices_df.query('slice_no>400').head(1).iterrows())[1]
b_img = imread(test_row['binned_images'])[::2, ::2, 0]
seg_img = imread(test_row['segmentation'])[::2, ::2, 0]
print('Image', b_img.shape, b_img.min(), b_img.max())
print('Segmentation', seg_img.shape, seg_img.min(), seg_img.max(), 'Unique Values', len(np.unique(seg_img)))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(b_img)
ax2.imshow(seg_img)
fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
ax1.imshow(montage_pad([imread(x)[::2, ::2, 0] for x in paired_slices_df['binned_images'].values[::8]]))
fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
ax1.imshow(montage_pad([imread(x)[::2, ::2, 0] for x in paired_slices_df['segmentation'].values[::8]]))
from scipy.ndimage import zoom
from skimage import measure
v = np.stack([imread(x)[::2, ::2, 0]>0 for x in paired_slices_df['segmentation'].values[::2]],0)
smooth_pt_img = zoom(v, (0.4, 0.4, 0.4), order = 3)
print(smooth_pt_img.shape)
verts, faces, _, _ = measure.marching_cubes_lewiner(
    smooth_pt_img, # you can make it bigger but the file-size gets HUUUEGE 
    smooth_pt_img.mean())
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
cmap = plt.cm.get_cmap('nipy_spectral_r')
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

mesh = Poly3DCollection(verts[faces], alpha=0.25, edgecolor='none', linewidth = 0.1)

mesh.set_edgecolor([1, 0, 0])
ax.add_collection3d(mesh)

ax.set_xlim(0, smooth_pt_img.shape[0])
ax.set_ylim(0, smooth_pt_img.shape[1])
ax.set_zlim(0, smooth_pt_img.shape[2])

ax.view_init(45, 45)
fig.savefig('couple.png')
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
py.init_notebook_mode()
smooth_pt_img = zoom(v, (0.33, 0.33, 0.33), order = 3)
print(smooth_pt_img.shape)
verts, faces, _, _ = measure.marching_cubes_lewiner(
    smooth_pt_img, # you can make it bigger but the file-size gets HUUUEGE 
    smooth_pt_img.mean())
x, y, z = zip(*verts)
ff_fig = FF.create_trisurf(x=x, y=y, z=z,
                           simplices=faces,
                           colormap=['rgb(255, 0, 0)', 'rgb(255, 0, 100)', ],
                           show_colorbar=True,
                           title="Segmentation",
                           aspectratio=dict(x=1, y=1, z=1),
                           plot_edges=False)
c_mesh = ff_fig['data'][0]
c_mesh.update(lighting=dict(ambient=0.18,
                            diffuse=1,
                            fresnel=0.1,
                            specular=1,
                            roughness=0.1,
                            facenormalsepsilon=1e-6,
                            vertexnormalsepsilon=1e-12))
c_mesh.update(flatshading=False)
py.iplot(ff_fig)
