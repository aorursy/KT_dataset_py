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
stack_image= stack_image[:,:,:]
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
water = 0
air = 0
for layer in bubble_inver:
    for i in layer:
        for j in i:
            if j == 0:
                water += 1
            else:
                air += 1
print("The liquid fraction is", water/air)
            
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt as distmap
bubble_dist = distmap(bubble_inver)
plt.imshow(bubble_dist[100,:,:], interpolation = 'none', cmap ='jet')
from skimage.feature import peak_local_max
bubble_candidates = peak_local_max(bubble_dist, min_distance=12)
print('Found',len(bubble_candidates), 'bubbles')
df = pd.DataFrame(data=bubble_candidates, columns=['x','y','z'])
df.to_csv('bubble.candidates.csv')
from skimage.morphology import watershed
bubble_seeds = peak_local_max(bubble_dist, min_distance=12, indices=False)
plt.imshow(np.sum(bubble_seeds,0).squeeze(), interpolation = 'none', cmap='bone_r')
markers = ndi.label(bubble_seeds)[0]
cropped_markers = markers[:,:,:]
cropped_bubble_dist=bubble_dist[:,:,:]
cropped_bubble_inver=bubble_inver[:,:,:]
labeled_bubbles= watershed(-cropped_bubble_dist, cropped_markers, mask=cropped_bubble_inver)
print(len(labeled_bubbles))
plt.imshow(labeled_bubbles[50,:,:], cmap=plt.cm.Spectral, interpolation='nearest')
from skimage.measure import regionprops
props=regionprops(labeled_bubbles)
props[20].filled_area
print(len(props))
filled = 0
endindex = len(props)
for i in range(0,endindex):
    filled += props[i].filled_area
print(filled)

bubble_volume=[prop.filled_area for prop in props]
bubble_volume_mean=np.mean(bubble_volume)
dfV = pd.DataFrame(data=bubble_volume, columns=['volume [pix^3]'])
dfV.to_csv('bubbleVolumes.csv')
Vm = {'mean volume': [1,bubble_volume_mean]}
dfVm=pd.DataFrame(data=Vm)
from tifffile import imsave
imsave('labeled_bubbles.tif', labeled_bubbles)
bub_eccentricity = [prop.extent for prop in props]
dfV = pd.DataFrame(data=bub_eccentricity, columns=['eccentricity'])
dfV.to_csv('bubbleEccentricities.csv')

bub_bbox = [prop.bbox for prop in props]
approx_area = [4 * np.pi * np.power((np.power(t[3] * t[3] / 4,1.6075) + np.power(t[3] * t[5] / 4,1.6075) + np.power(t[4] * t[5] / 4,1.6075)) / 3,1/1.6075) for t in bub_bbox]
dfV = pd.DataFrame(data=approx_area, columns=['area [pix^2]'])
dfV.to_csv('bubbleArea.csv')

