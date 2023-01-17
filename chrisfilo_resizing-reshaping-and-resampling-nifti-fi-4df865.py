from nilearn.image import resample_img
import pylab as plt
import nibabel as nb
import numpy as np
orig_nii = nb.load("../input/ixi-example/ixi002-guys-0828-t1.nii/IXI002-Guys-0828-T1.nii")
np.round(orig_nii.affine)
orig_nii.shape
orig_nii.header.get_zooms()
plt.imshow(orig_nii.dataobj[:,:,80])
orig_rotated_nii = nb.as_closest_canonical(orig_nii)
np.round(orig_rotated_nii.affine)
orig_rotated_nii.shape
plt.imshow(orig_rotated_nii.dataobj[:,:,80])
downsampled_nii = resample_img(orig_rotated_nii, target_affine=np.eye(3)*2., interpolation='nearest')
downsampled_nii.affine
downsampled_nii.shape
plt.imshow(downsampled_nii.dataobj[:,:,50])
upsampled_nii = resample_img(orig_rotated_nii, target_affine=np.eye(3)*0.5, interpolation='nearest')
upsampled_nii.affine
upsampled_nii.shape
plt.imshow(upsampled_nii.dataobj[:,:,200])
target_shape = np.array((240,40,100))
new_resolution = [2,]*3
new_affine = np.zeros((4,4))
new_affine[:3,:3] = np.diag(new_resolution)
# putting point 0,0,0 in the middle of the new volume - this could be refined in the future
new_affine[:3,3] = target_shape*new_resolution/2.*-1
new_affine[3,3] = 1.
downsampled_and_cropped_nii = resample_img(orig_rotated_nii, target_affine=new_affine, target_shape=target_shape, interpolation='nearest')
downsampled_and_cropped_nii.affine
downsampled_and_cropped_nii.shape
plt.imshow(downsampled_and_cropped_nii.dataobj[:,:,70])
