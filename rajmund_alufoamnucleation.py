# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from skimage.io import imread, imsave

from glob import glob

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm # a nice progress bar
stack_image_o=imread('../input/098_rec06881_stack.tif')

stack_image=stack_image_o[:,30:470,30:470]
nhight, ncols, nrows = stack_image.shape

row, col = np.ogrid[:nrows, :ncols]

print(stack_image.shape,stack_image.dtype)
%matplotlib inline

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):

    cax.imshow(np.sum(stack_image,i).squeeze(), interpolation='none', cmap = 'bone_r')

    cax.set_title('%s Projection' % clabel)

    cax.set_xlabel(clabel[0])

    cax.set_ylabel(clabel[1])
plt.imshow(stack_image[100],cmap='bone')
from scipy import stats

vmin, vmax = stats.scoreatpercentile(stack_image, (0.5, 99.5))
stack_image = np.clip(stack_image, vmin, vmax)

stack_image = (stack_image - vmin) / (vmax - vmin)

plt.imshow(stack_image[60], cmap='gray')
from skimage import restoration

tv = restoration.denoise_tv_chambolle(stack_image[:,:,:],weight=0.2)

plt.imshow(tv[60], cmap='gray')
from skimage import exposure

hi_stack_image = exposure.histogram(stack_image)

plt.plot(hi_stack_image[1], hi_stack_image[0], label='data')

plt.xlim(0, 1)

plt.legend()

plt.title('Histogram of voxel values')
hi_tv=exposure.histogram(tv)

plt.plot(hi_tv[1], hi_tv[0])
seg_stack_image = stack_image >0.4

seg_stack_image_inv=np.invert(seg_stack_image)

plt.imshow(seg_stack_image_inv[60],cmap='gray')
stack_image_s=stack_image[50:70,:,:]

markers = np.zeros(stack_image.shape, dtype=np.uint8)

markers[stack_image > 0.55] = 1

markers[stack_image < 0.45] = 2

plt.imshow(markers[60], cmap='gray')
from skimage import segmentation

stack_rw = segmentation.random_walker(stack_image, markers, beta=1000., mode='cg_mg')
plt.imshow(stack_image[10], cmap='gray')

plt.contour(stack_rw[10], [1.5])
from tifffile import imsave

#image = np.zeros((32, 256, 256), 'uint16')

imsave('stack_rw.tif', stack_rw)
plt.imshow(np.invert(stack_rw[60])==-2)
from skimage.morphology import binary_opening, convex_hull_image as chull

#seg_stack_image_inv_s=seg_stack_image_inv[50:70,:,:]

#bubble_image = np.stack([chull(csl>0) & (csl==0) for csl in stack_rw])

bubble_image = np.stack([chull(csl<2) & (csl==1) for csl in stack_rw])

plt.imshow(bubble_image[60]>0, cmap = 'bone')
bubble_invert=np.invert(bubble_image)

plt.imshow(bubble_invert[10],cmap='bone')
from scipy import ndimage as ndi

from scipy.ndimage.morphology import distance_transform_edt as distmap

bubble_dist=distmap(bubble_invert)
plt.imshow(bubble_dist[60,:,:],interpolation='none',cmap='jet')
from skimage.feature import peak_local_max

bubble_candidates=peak_local_max(bubble_dist,min_distance=6)

print('Found',len(bubble_candidates),'bubbles')
df = pd.DataFrame(data=bubble_candidates, columns=['x','y','z'])

df.to_csv('./bubble.candidates.csv')
from skimage.morphology import watershed

bubble_seeds=peak_local_max(bubble_dist,min_distance=4, indices='false')

plt.imshow(np.sum(bubble_seeds,0).squeeze(),interpolation='none', cmap='bone_r')
markers = ndi.label(bubble_seeds)[0]

labeled_bubbles= watershed(-bubble_dist, markers, mask=bubble_invert)
plt.imshow(labeled_bubbles[60,:,:], plt.cm.nipy_spectral, interpolation='nearest')
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

ax1.imshow(labeled_bubbles[15,:,:], plt.cm.nipy_spectral, interpolation='nearest')

ax2.imshow(labeled_bubbles[:,250,:], plt.cm.nipy_spectral, interpolation='nearest')

ax3.imshow(labeled_bubbles[:,:,250], plt.cm.nipy_spectral, interpolation='nearest')
from skimage.measure import regionprops

from skimage.measure import marching_cubes_lewiner

from skimage.measure import mesh_surface_area

regions=regionprops(labeled_bubbles)

regions[40].filled_area

regions[40].centroid
bubbles_mc=marching_cubes_lewiner(labeled_bubbles, level=None, spacing=(1.0, 1.0, 1.0), gradient_direction='descent', step_size=1, allow_degenerate=True, use_classic=False)
bubble_surfaces=mesh_surface_area(bubbles_mc[0],bubbles_mc[1])

print(bubble_surfaces)
from tifffile import imsave

#image = np.zeros((32, 256, 256), 'uint16')

imsave('labeled.tif', labeled_bubbles)
bubble_volume=[prop.filled_area for prop in regions]

bubble_volume_mean=np.mean(bubble_volume)

dfV = pd.DataFrame(data=bubble_volume, columns=['volume [pix^3]'])

dfV.to_csv('AlububbleVolumes.csv')

Vm = {'mean volume': [1,bubble_volume_mean]}

dfVm=pd.DataFrame(data=Vm)
bubble_diameter=[prop.equivalent_diameter for prop in regions]

bubble_diameter_mean=np.mean(bubble_diameter)

print('mean bubble diamter [pixels]',bubble_diameter_mean)

dfD=pd.DataFrame(data=bubble_diameter, columns=['diameter [pix]'])

dfD.to_csv('AluBubbleDiameters.csv')

Dm={'mean diameter':[1,bubble_diameter_mean]}

dfDm=pd.DataFrame(data=Dm)
fig, (ax1,ax2,ax3)=plt.subplots(1,3,figsize=(20,5))

ax1.hist(bubble_volume,bins=100,range= [80,2000])

ax1.set_title('volume distribution')

ax2.hist(bubble_diameter,np.linspace(2,100,50),label='sample1mid_t0')

ax2.set_title('radii distribution')
thickness_map = np.zeros(bubble_dist.shape, dtype = np.float32)

xx, yy, zz = np.meshgrid(np.arange(bubble_dist.shape[1]),

                         np.arange(bubble_dist.shape[0]),

                         np.arange(bubble_dist.shape[2])

                        )

# sort candidates by size

sorted_candidates = sorted(bubble_candidates, key = lambda xyz: bubble_dist[tuple(xyz)])

for label_idx, (x,y,z) in enumerate(tqdm(sorted_candidates),1):

    cur_bubble_radius = bubble_dist[x,y,z]

    cur_bubble = (np.power(xx-float(y),2)+

                  np.power(yy-float(x),2)+

                  np.power(zz-float(z),2))<=np.power(cur_bubble_radius,2)

    thickness_map[cur_bubble] = cur_bubble_radius
%matplotlib inline

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):

    cax.imshow(np.max(thickness_map,i).squeeze(), interpolation='none', cmap = 'jet')

    cax.set_title('%s Projection' % clabel)

    cax.set_xlabel(clabel[0])

    cax.set_ylabel(clabel[1])
imsave('AluFoamThickMap.tif',thickness_map)
from mayavi import mlab

mlab.contour3d(stack_rw)