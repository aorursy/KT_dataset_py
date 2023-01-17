from skimage.io import imread, imsave

from glob import glob

import numpy as np

import matplotlib.pyplot as plt
stack_image = imread('../input/plateau_border.tif')

print(stack_image.shape, stack_image.dtype)
%matplotlib inline

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):

    cax.imshow(np.sum(stack_image,i).squeeze(), interpolation='none', cmap = 'bone_r')

    cax.set_title('%s Projection' % clabel)

    cax.set_xlabel(clabel[0])

    cax.set_ylabel(clabel[1])
from skimage.morphology import binary_opening, convex_hull_image as chull

bubble_image = np.stack([chull(csl>0) & (csl==0) for csl in stack_image])

plt.imshow(bubble_image[5]>0, cmap = 'bone')
%%time

from scipy.ndimage.morphology import distance_transform_edt as distmap

bubble_dist = distmap(bubble_image)
%matplotlib inline

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

ax1.imshow(bubble_dist[100], interpolation='none', cmap = 'jet')

ax1.set_title('YZ Slice')

ax2.imshow(bubble_dist[:,100], interpolation='none', cmap = 'jet')

ax2.set_title('XZ Slice')

ax3.imshow(bubble_dist[:,:,100], interpolation='none', cmap = 'jet')

ax3.set_title('XY Slice')

%matplotlib inline

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):

    cax.imshow(np.max(bubble_dist,i).squeeze(), interpolation='none', cmap = 'magma')

    cax.set_title('%s Projection' % clabel)

    cax.set_xlabel(clabel[0])

    cax.set_ylabel(clabel[1])
from skimage.feature import peak_local_max

bubble_candidates = peak_local_max(bubble_dist, min_distance=20)

print('Found', len(bubble_candidates), 'bubbles')

bubble_labels = np.zeros(bubble_dist.shape, dtype = np.int16)

# sort candidates by size

sorted_candidates = sorted(bubble_candidates, key = lambda xyz: bubble_dist[tuple(xyz)])

for label_idx, (x,y,z) in enumerate(sorted_candidates,1):

    bubble_labels[x,y,z] = label_idx
%%time

from skimage.morphology import ball

from scipy import ndimage as ndi

base_image = bubble_labels.copy()

# run iteratively and only replace empty pixels

for cur_rad in range(10, 120, 10):

    latest_filling = ndi.maximum_filter(bubble_labels, cur_rad)

    new_val_mask = (base_image == 0) & bubble_image & (latest_filling>0)

    base_image[new_val_mask] = latest_filling[new_val_mask]

bubble_label_image = base_image
# reorder the labels to make it easier to see

new_label_image = np.zeros_like(bubble_label_image)

label_idxs = [i for i in np.random.permutation(np.unique(bubble_label_image)) if i>0]

for new_label, old_label in enumerate(label_idxs):

    new_label_image[bubble_label_image==old_label] = new_label

%matplotlib inline

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):

    cax.imshow(np.max(new_label_image,i).squeeze(), interpolation='none', cmap = 'gist_earth')

    cax.set_title('%s Projection' % clabel)

    cax.set_xlabel(clabel[0])

    cax.set_ylabel(clabel[1])
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure

from tqdm import tqdm

def show_3d_mesh(image, thresholds):

    p = image[::-1].swapaxes(1,2)

    cmap = plt.cm.get_cmap('nipy_spectral_r')

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    for i, c_threshold in tqdm(list(enumerate(thresholds))):

        verts, faces = measure.marching_cubes(p==c_threshold, 0)

        mesh = Poly3DCollection(verts[faces], alpha=0.25, edgecolor='none', linewidth = 0.1)

        mesh.set_facecolor(cmap(i / len(thresholds))[:3])

        mesh.set_edgecolor([1, 0, 0])

        ax.add_collection3d(mesh)



    ax.set_xlim(0, p.shape[0])

    ax.set_ylim(0, p.shape[1])

    ax.set_zlim(0, p.shape[2])

    

    ax.view_init(45, 45)

    return fig
_ = show_3d_mesh(new_label_image, label_idxs[:15])
def meshgrid3d_like(in_img):

    return np.meshgrid(range(in_img.shape[1]),range(in_img.shape[0]), range(in_img.shape[2]))

zz, xx, yy = meshgrid3d_like(bubble_label_image)
out_results = []

for c_label in np.unique(bubble_label_image): # one bubble at a time

    if c_label>0: # ignore background

        cur_roi = bubble_label_image==c_label

        out_results += [{'x': xx[cur_roi].mean(), 'y': yy[cur_roi].mean(), 'z': zz[cur_roi].mean(), 

                         'volume': np.sum(cur_roi)}]
import pandas as pd

out_table = pd.DataFrame(out_results)

out_table.to_csv('bubble_volume.csv')

out_table.sample(5)
out_table['volume'].plot.density()
out_table.plot.hexbin('x', 'y', gridsize = (5,5))
train_values = pd.read_csv('../input/bubble_volume.csv')
%matplotlib inline

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))

ax1.hist(np.log10(out_table['volume']))

ax1.hist(np.log10(train_values['volume']))

ax1.legend(['Distance Map Volumes', 'Training Volumes'])

ax1.set_title('Volume Comparison\n(Log10)')

ax2.plot(out_table['x'], out_table['y'], 'r.',

        train_values['x'], train_values['y'], 'b.')

ax2.legend(['Distance Map Bubbles', 'Training Bubbles'])