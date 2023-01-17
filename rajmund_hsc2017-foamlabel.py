# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from skimage.io import imread, imsave

from glob import glob

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm # a nice progress bar

import pandas as pd
stack_image = imread('../input/rec_8bit_ph03_cropC_kmeans_scale510.tif')

print(stack_image.shape, stack_image.dtype)
%matplotlib inline

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):

    cax.imshow(np.sum(stack_image,i).squeeze(), interpolation='none', cmap = 'bone_r')

    cax.set_title('%s Projection' % clabel)

    cax.set_xlabel(clabel[0])

    cax.set_ylabel(clabel[1])
plt.imshow(stack_image[100],cmap='bone') # showing slice No.100
from skimage.morphology import binary_opening, convex_hull_image as chull

bubble_image = np.stack([chull(csl>0) & (csl==0) for csl in stack_image])

plt.imshow(bubble_image[5]>0, cmap = 'bone')
bubble_inver=np.invert(bubble_image)
plt.imshow(bubble_inver[100], cmap='bone')
from scipy import ndimage as ndi

from scipy.ndimage.morphology import distance_transform_edt as distmap

bubble_dist = distmap(bubble_inver)
plt.imshow(bubble_dist[200,:,:], interpolation = 'none', cmap ='jet')
from skimage.feature import peak_local_max

bubble_candidates = peak_local_max(bubble_dist, min_distance=12)

print('Found',len(bubble_candidates), 'bubbles')
df = pd.DataFrame(data=bubble_candidates, columns=['x','y','z'])

df.to_csv('bubble.candidates.csv')
from skimage.morphology import watershed

bubble_seeds = peak_local_max(bubble_dist, min_distance=12, indices=False)

plt.imshow(np.sum(bubble_seeds,0).squeeze(), interpolation = 'none', cmap='bone_r')
markers = ndi.label(bubble_seeds)[0]

cropped_markers = markers[200:300,100:400,100:400]

cropped_bubble_dist=bubble_dist[200:300,100:400,100:400]

cropped_bubble_inver=bubble_inver[200:300,100:400,100:400]

labeled_bubbles= watershed(-cropped_bubble_dist, cropped_markers, mask=cropped_bubble_inver)
plt.imshow(labeled_bubbles[50,:,:], cmap=plt.cm.Spectral, interpolation='nearest')
from skimage.measure import regionprops

regions=regionprops(labeled_bubbles)

regions[20].filled_area
bubble_volume=[prop.filled_area for prop in regions]

bubble_volume_mean=np.mean(bubble_volume)

dfV = pd.DataFrame(data=bubble_volume, columns=['volume [pix^3]'])

dfV.to_csv('bubbleVolumes.csv')

Vm = {'mean volume': [1,bubble_volume_mean]}

dfVm=pd.DataFrame(data=Vm)
from tifffile import imsave

imsave('labeled_bubbles.tif',labeled_bubbles)
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