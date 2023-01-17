%matplotlib inline

import nibabel

import matplotlib.pyplot as plt

import numpy as np
# Load image files with nibabel 

CT = nibabel.load('../input/covid19-ct-scans/ct_scans/coronacases_org_001.nii')
CT_array = CT.get_fdata()

CT_array.shape
CT_array = CT_array.T

plt.imshow(CT_array[200])
mask = nibabel.load('../input/covid19-ct-scans/lung_mask/coronacases_001.nii')



left_mask_array = mask.get_fdata()

left_mask_array = left_mask_array.T

right_mask_array = left_mask_array.copy()



plt.imshow(left_mask_array[200])
left_mask_array[left_mask_array == 2] = 0

right_mask_array[right_mask_array == 1] = 0

right_mask_array = right_mask_array / 2



left_lung_array = np.multiply(CT_array, left_mask_array)

right_lung_array = np.multiply(CT_array, right_mask_array)



left_lung_array = left_lung_array - np.amin(left_lung_array)

left_lung_array = left_lung_array / np.amax(left_lung_array)



right_lung_array = right_lung_array - np.amin(right_lung_array)

right_lung_array = right_lung_array / np.amax(right_lung_array)



plt.imshow(left_lung_array[200], cmap='jet')
width = left_mask_array.shape[1]

height = left_mask_array.shape[2]



xmin = 0

xmax = width

ymin = 0

ymax = height



for i in range(round(width / 2)):

    if np.sum(left_mask_array[:,:i,:]) == 0: xmin = i

    if np.sum(left_mask_array[:,width - i:,:]) == 0: xmax = width - i

    if np.sum(left_mask_array[:,:,:i]) == 0: ymin = i

    if np.sum(left_mask_array[:,:,height - i:]) == 0: ymax = height - i

    

print(xmin, xmax, ymin, ymax)

left_lung_array = left_lung_array[:,xmin:xmax,ymin:ymax]



plt.imshow(left_lung_array[200], cmap='jet')