%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.tri as mtri

from IPython.display import Image, display, SVG, clear_output, HTML

plt.rcParams["figure.figsize"] = (6, 6)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

plt.rcParams['image.cmap'] = 'gray' # grayscale looks better
import numpy as np # linear algebra

from PIL import Image as PImage

# opencv and skimage don't support it yet

read_compressed_tiff = lambda x: PImage.open(x)

from skimage.util import montage as montage2d

import pandas as pd

from pathlib import Path

from tqdm import tqdm

tqdm.pandas()
tiff_root = Path('..') / 'input' / 'evolutionary-origins-of-teeth'

all_tiff_df = pd.DataFrame({'path': list(tiff_root.glob('**/*.tif'))})

all_tiff_df['full_id'] = all_tiff_df['path'].map(lambda x: x.relative_to(tiff_root).parent)

all_tiff_df['subject'] = all_tiff_df['full_id'].map(lambda x: x.parts[0])

all_tiff_df['sample'] = all_tiff_df['full_id'].map(lambda x: x.parts[1])

all_tiff_df['specimen'] = all_tiff_df['path'].map(lambda x: x.stem[:-4])

all_tiff_df['slice'] = all_tiff_df['path'].map(lambda x: int(x.stem[-4:]))

all_tiff_df.sample(5)
all_tiff_df.groupby(['full_id']).apply(lambda x: x.shape[0]).reset_index()
sample_row = all_tiff_df.sample(1).iloc[0]

print(sample_row)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

sample_img = read_compressed_tiff(sample_row['path'])

ax1.imshow(sample_img)

ax1.set_title(sample_row['full_id'])

ax2.hist(np.ravel(sample_img))

ax2.set_yscale('log')
cur_slice_df = all_tiff_df[all_tiff_df['full_id'].isin({sample_row['full_id']})].sort_values('slice')

cur_slice_df.head(3)
fossil_data = np.stack(cur_slice_df.iloc[::2]['path'].progress_map(lambda path: 

                                                                   np.array(read_compressed_tiff(path))[::2, ::2]), 0)



print('Loading Fossil Data sized {}'.format(fossil_data.shape))
%matplotlib inline

slice_idx = int(fossil_data.shape[0]/2)

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 5))



ax1.imshow(fossil_data[slice_idx], cmap = 'bone')

ax1.set_title('Axial Slices')

_ = ax2.hist(fossil_data[slice_idx].ravel(), 20)

ax2.set_title('Slice Histogram')

ax2.set_yscale('log')
%matplotlib inline

from scipy.ndimage.filters import median_filter

# filter the data

filter_fossil_data = median_filter(fossil_data, (3, 3, 3))



# setup the plot

slice_idx = int(fossil_data.shape[0]/2)

test_slice = fossil_data[slice_idx]

test_filt_slice = filter_fossil_data[slice_idx]

# setup the default image arguments

im_args = dict(cmap = 'bone', vmin = 0, vmax = 60)



fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 5))

ax1.imshow(test_slice, **im_args)

ax1.set_title('Unfiltered Slice')

_ = ax2.imshow(test_filt_slice, **im_args)

ax2.set_title('Filtered Slice')
%matplotlib inline

skip_border = 50

skip_middle = 4

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (14, 5))

ax1.imshow(montage2d(filter_fossil_data[skip_border:-skip_border:skip_middle]),**im_args)

ax1.set_title('Axial Slices')

ax1.axis('off')



ax2.imshow(montage2d(filter_fossil_data.transpose(1,2,0)[skip_border:-skip_border:skip_middle]), **im_args)

ax2.set_title('Saggital Slices')

ax2.axis('off')



ax3.imshow(montage2d(filter_fossil_data.transpose(2,0,1)[skip_border:-skip_border:skip_middle]), **im_args)

ax3.set_title('Coronal Slices')

ax3.axis('off')
from skimage.filters import try_all_threshold

try_all_threshold(filter_fossil_data[slice_idx])
%matplotlib inline

from skimage.filters import threshold_yen

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 5))

thresh_fossil_data = filter_fossil_data > threshold_yen(filter_fossil_data)

thresh_slice = thresh_fossil_data[slice_idx]

ax1.imshow(test_filt_slice, cmap = 'bone')

ax1.set_title('Filtered Slices')

_ = ax2.imshow(thresh_slice)

ax2.set_title('Slice with Threshold')
%matplotlib inline

from skimage.morphology import binary_closing, ball, binary_opening

closed_fossil_data = binary_closing(thresh_fossil_data, ball(5))

closed_fossil_data = binary_opening(closed_fossil_data, ball(3))

close_slice = closed_fossil_data[slice_idx]

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 5))



ax1.imshow(test_filt_slice, cmap = 'bone')

ax1.set_title('Filtered Slices')

_ = ax2.imshow(close_slice)

ax2.set_title('Slice After Closing')
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure

def show_3d_mesh(p, threshold):

    verts, faces, _, _ = measure.marching_cubes_lewiner(p, threshold)



    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.9, edgecolor='none', linewidth = 0.1)

    mesh.set_facecolor([.1, 1, .1])

    mesh.set_edgecolor([1, 0, 0])

    

    ax.add_collection3d(mesh)



    ax.set_xlim(0, p.shape[0])

    ax.set_ylim(0, p.shape[1])

    ax.set_zlim(0, p.shape[2])

    

    ax.view_init(45, 45)

    return fig
from scipy.ndimage import zoom

# we downsample the image to make 3d rendering quicker

fossil_downscale = zoom(closed_fossil_data.astype(np.float32), 0.5)

# now we display it with a nice mesh

_ = show_3d_mesh(fossil_downscale, 0.5)