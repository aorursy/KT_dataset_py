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
from skimage.morphology import ball as skm_ball, binary_erosion

bubble_centers = binary_erosion(bubble_image[:,::3, ::3], selem = skm_ball(5))
%matplotlib inline

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):

    cax.imshow(np.sum(bubble_centers,i).squeeze(), interpolation='none', cmap = 'bone_r')

    cax.set_title('%s Projection' % clabel)

    cax.set_xlabel(clabel[0])

    cax.set_ylabel(clabel[1])
from skimage.morphology import label

bubble_center_label_image = label(bubble_centers)
%matplotlib inline

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):

    cax.imshow(np.max(bubble_center_label_image,i).squeeze(), interpolation='none', cmap = 'jet')

    cax.set_title('%s Projection' % clabel)

    cax.set_xlabel(clabel[0])

    cax.set_ylabel(clabel[1])
from skimage.morphology import dilation

from scipy import ndimage

bubble_label_image = dilation(bubble_center_label_image, skm_ball(5))

bubble_label_image = ndimage.zoom(bubble_label_image, (1, 3, 3), order = 0) # nearest neighbor

%matplotlib inline

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):

    cax.imshow(np.max(bubble_label_image,i).squeeze(), interpolation='none', cmap = 'jet')

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
_ = show_3d_mesh(bubble_label_image, range(1,np.max(bubble_label_image), 10))
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

ax1.legend(['Erosion Volumes', 'Training Volumes'])

ax1.set_title('Volume Comparison\n(Log10)')

ax2.plot(out_table['x'], out_table['y'], 'r.',

        train_values['x'], train_values['y'], 'b.')

ax2.legend(['Erosion Bubbles', 'Training Bubbles'])