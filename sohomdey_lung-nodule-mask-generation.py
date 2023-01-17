import os
import glob
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SimpleITK as sitk
def get_nodule_mask(file_path, annot_df):
    
    '''
    input:
        file_path: image file path (for which mask is to be generated)
        annot_df: pandas dataframe (annotations.csv file loaded)
    
    output:
        nodule mask in SimpleITK format
    '''

    # Load the image
    image_sitk = sitk.ReadImage(file_path)
    
    # Get the image dimension
    image_dims = image_sitk.GetSize()

    # Get the origin and spacing of the image
    origin = np.array(image_sitk.GetOrigin())
    spacing = np.array(image_sitk.GetSpacing())

    # Get the file id from the file path (without the extention)
    file_id = file_path.split('/')[-1][:-4]
    
    # Get the annotations (coordX, coordY, coordZ, diameter_mm)
    annotations = annot_df[annot_df['seriesuid']==file_id].values[:, 1:].astype(np.float32)

    # Create a all-zero mask of the same shape as the image
    mask_array = np.zeros(shape=image_dims, dtype=np.uint8)

    # Check if there are any nodule annotation available for the image
    if len(annotations) != 0:

        # Iterate over all the available nodule annotations
        for node in annotations:

            # Extract the center and diameter
            center = node[0:3]
            diamtr = node[3]

            # Convert the center and diameter from physical space to array 
            center = np.rint((center - origin) / spacing)
            diamtr = (diamtr/spacing + 1)
            
            # Change the order of the center elements for array row/column calculations 
            center = (center[1], center[0], center[2])
            # Get the radius
            radius = np.rint(diamtr/2)

            # Get the coordinates for the nodule volume (cuboid)
            x_min, y_min, z_min = (center-radius).astype(int)
            x_max, y_max, z_max = (center+radius).astype(int)
            
            # Set the nodule volume (cuboid) in the mask as 1
            mask_array[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = 1

    # Change the arrangement of the axes so that they concide with that of the image
    mask_array = np.transpose(mask_array, (2,0,1))

    # Change the mask format from numpy to SimpleITK
    mask_sitk = sitk.GetImageFromArray(mask_array)
    
    # Set the origin and spacing to that of the image
    mask_sitk.SetSpacing(spacing)
    mask_sitk.SetOrigin(origin)

    return mask_sitk
# Create a mask folder to store all the masks

if os.path.exists('/kaggle/working/mask'):
    shutil.rmtree('/kaggle/working/mask')
    
os.mkdir('/kaggle/working/mask')
# Set the image directory
img_dir = '/kaggle/input/luna16p1'
# Set the mask directory (where masks will be saved)
msk_dir = '/kaggle/working/mask'

# Load the annotations in a pandas dataframe
annot_df = pd.read_csv('/kaggle/input/luna16p1/annotations.csv')

# Check how many image subsets are there in te image directory
# n_subset = len(glob.glob(f'{img_dir}/*subset[0-9]'))

n_subset = len(glob.glob(f'{img_dir}/*subset[0-9]'))

# Iterate over all the subsets
for i in range(n_subset):
    
    # Subset name
    subset = f'subset{i}'
    
    # Set the image and mask sub directories (may need to be changed depending on the folder arrangement)
    img_subset_dir = f'{img_dir}/{subset}/{subset}'
    msk_subset_dir = f'{msk_dir}/{subset}/{subset}'
    
    # Create the sub directory in the mask folder
    os.makedirs(msk_subset_dir)
    
    # Get all the image files in the image sub directory
    img_files = sorted(glob.glob(f'{img_subset_dir}/*.mhd'))[:2]
    
    # Iterate over all the image files
    for file_path in img_files:

        # Get the mask (SimpleITK format)
        mask_sitk = get_nodule_mask(file_path, annot_df)

        # Get the file id from the file path (without the extention)
        file_id = file_path.split('/')[-1][:-4]
        # Set the path where mask is to be saved
        write_path = f'{msk_subset_dir}/{file_id}.mhd'

        # Write the image
        sitk.WriteImage(mask_sitk, write_path)
        
    # Print when a subset is complete
    print(f'{subset} complete')
def resample_sitk(image_sitk, mask_sitk):
    
    '''
    input:
        image_sitk: image in SimpleITK format
        mask_sitk: mask in SimpleITK format
    
    output:
        Resampled image in SimpleITK format
    '''
    
    # Get the original dimensions
    image_dims = image_sitk.GetSize()
    
    # Get the original spacing
    spacing = np.array(image_sitk.GetSpacing())

    # New spacing 
    new_spacing = np.array([spacing[0], spacing[1], 1])
    
    # Resampling factor
    factor = spacing/new_spacing
    
    # Calculate the new dimensions
    new_image_dims = (image_dims*factor).astype(np.int)

    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_sitk)
    resampler.SetSize(new_image_dims.tolist())
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_img_sitk = resampler.Execute(image_sitk)
    
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_msk_sitk = resampler.Execute(mask_sitk)
    
    return resampled_img_sitk, resampled_msk_sitk
def get_roi_img_msk(image_sitk, mask_sitk, expandslice=10):
    
    '''
    input:
        image_sitk: image in SimpleITK format (DxWxH)
        mask_sitk: mask in SimpleITK format (DxWxH)
        expandslice: number of extra slices to include in each direction 
    
    output:
        roi_img_sitk: image with fewer slices which contains the nodule 
        roi_msk_sitk: mask with fewer slices which contains the nodule 
    '''
    
    # Get numpy array
    img_array = sitk.GetArrayFromImage(image_sitk)
    msk_array = sitk.GetArrayFromImage(mask_sitk)
    
    
    # Get the slice indices which contains the nodules 
    roi_slice_indices = np.sort(np.where(msk_array)[0])

    # If there are nodules get the roi
    if len(roi_slice_indices) != 0:
        
        # Get the start and end indices of the roi slices 
        roi_idx = roi_slice_indices[[0, -1]]
        
        # Include extra slices in both directions
        roi_idx[0] -= expandslice
        roi_idx[1] += expandslice

        # Make sure the indices value stays between 0 and no. of slices
        n_slices = msk_array.shape[0]
        roi_idx = np.clip(roi_idx, 0, n_slices)

        # Extract the roi slices
        roi_img = img_array[roi_idx[0]:roi_idx[1],:,:]
        roi_msk = msk_array[roi_idx[0]:roi_idx[1], :,:]
        
        # Convert back to SimpleITK
        roi_img_sitk = sitk.GetImageFromArray(roi_img)
        roi_msk_sitk = sitk.GetImageFromArray(roi_msk)
        
        # Get the origin and spacing
        origin = np.array(image_sitk.GetOrigin())
        spacing = np.array(image_sitk.GetSpacing())
    
        # Set the origin and spacing
        roi_img_sitk.SetSpacing(spacing)
        roi_img_sitk.SetOrigin(origin)
        roi_msk_sitk.SetSpacing(spacing)
        roi_msk_sitk.SetOrigin(origin)
        
    # If there are no nodules return the same image and mask
    else:
        
        roi_img_sitk = image_sitk
        roi_msk_sitk = mask_sitk
        
    return roi_img_sitk, roi_msk_sitk
# Create a roi folder to store all the roi images and masks

if os.path.exists('/kaggle/working/roi'):
    shutil.rmtree('/kaggle/working/roi')
    
os.makedirs('/kaggle/working/roi/image')
os.makedirs('/kaggle/working/roi/mask')
# Set the image directory
img_dir = '/kaggle/input/luna16p1'
# Set the mask directory
msk_dir = '/kaggle/working/mask'
n_subset = len(glob.glob(f'{msk_dir}/*subset[0-9]'))

# Iterate over all the subsets
for i in range(n_subset):
    
    # Subset name
    subset = f'subset{i}'
    
    # Set the image and mask sub directories (may need to be changed depending on the folder arrangement)
    img_subset_dir = f'{img_dir}/{subset}/{subset}'
    msk_subset_dir = f'{msk_dir}/{subset}/{subset}'
    
    # Create the sub directory in the roi folder
    os.makedirs(f'roi/image/{subset}/{subset}')
    os.makedirs(f'roi/mask/{subset}/{subset}')
    
    # Get all the image and mask files in the sub directories
    msk_files = sorted(glob.glob(f'{msk_subset_dir}/*.mhd'))
    img_files = [i.replace(msk_dir, img_dir) for i in msk_files]

    for img_file, msk_file in zip(img_files, msk_files):

        # Read the SimpleITK image
        img_sitk = sitk.ReadImage(img_file)
        msk_sitk = sitk.ReadImage(msk_file)

        # Resample the SimpleITK image to have 1mm spacing in depth direction
        resampled_img_sitk, resampled_msk_sitk = resample_sitk(img_sitk, msk_sitk)

        # Get the roi slices having nodules
        roi_img_sitk, roi_msk_sitk = get_roi_img_msk(resampled_img_sitk, resampled_msk_sitk)

        # Get the file id from the file path (without the extention)
        file_id = img_file.split('/')[-1][:-4]

        # Set the path where image and mask are to be saved
        img_write_path = f'roi/image/{subset}/{subset}/{file_id}.mhd'
        msk_write_path = f'roi/mask/{subset}/{subset}/{file_id}.mhd'
        
        # Write the image and mask
        sitk.WriteImage(roi_img_sitk, img_write_path)
        sitk.WriteImage(roi_msk_sitk, msk_write_path)

    # Print when a subset is complete
    print(f'{subset} complete')
def read_mhd(file_path):
    array = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(array)
    array = np.transpose(array, (1,2,0))
    return array
def clip_normalize(image, low=-700, high=-600):
    img = np.clip(image, low, high)
    img = (img - low) / (high - low)
    img = (img*255).astype(np.uint8)
    return img
def get_patches(img_array, msk_array, patch_volume=(128,128,16)):

    # Get dimensions
    image_h, image_w, image_d = img_array.shape
    patch_h, patch_w, patch_d = patch_volume
    
    # Get padding values
    pad_h = (patch_h//2 - image_h % patch_h//2) % patch_h//2
    pad_w = (patch_w//2 - image_w % patch_w//2) % patch_w//2
    pad_d = (patch_d//2 - image_d % patch_d//2) % patch_d//2
    paddings = [[0, pad_h], [0, pad_w], [0, pad_d]]
    
    # Padd the image and mask by the same amount
    img_array = np.pad(img_array, paddings, 'constant')
    msk_array = np.pad(msk_array, paddings, 'constant')

    # Array to save the patches
    img_patches = []
    msk_patches = []

    # Generate patches:
    i = 0
    while i < image_h:
        j = 0
        while j < image_w:
            k = 0
            while k < image_d:
                
                img_patch = img_array[i:i+patch_h, j:j+patch_w, k:k+patch_d]
                msk_patch = msk_array[i:i+patch_h, j:j+patch_w, k:k+patch_d]

                if img_patch.shape == patch_volume:
                    img_patches.append(img_patch)
                    msk_patches.append(msk_patch)

                k += patch_d // 2 
            j += patch_w // 2
        i += patch_h // 2 

    return np.array(img_patches), np.array(msk_patches)
def save_patches(img_patches, msk_patches, save_path):
    
    c = 0
    
    # Make the images and masks folder
    os.mkdir(f'{save_path}/images')
    os.mkdir(f'{save_path}/masks')
    
    for img_patch, msk_patch in zip(img_patches, msk_patches):

        # Convert to SimpleITK format 
        img_patch_sitk = sitk.GetImageFromArray(img_patch)
        msk_patch_sitk = sitk.GetImageFromArray(msk_patch)
     
        # Set the path where the patches are to be saved
        img_patch_write_path = f'{save_path}/images/{c:05}.mhd'
        msk_patch_write_path = f'{save_path}/masks/{c:05}.mhd'

        # Write the patches
        sitk.WriteImage(img_patch_sitk, img_patch_write_path)
        sitk.WriteImage(msk_patch_sitk, msk_patch_write_path)
        
        c+=1

img_dir = '/kaggle/working/roi/image'
msk_dir = '/kaggle/working/roi/mask'

patch_dir = '/kaggle/working/patches'
# Create a patches folder to store all the patches

if os.path.exists(patch_dir):
    shutil.rmtree(patch_dir)
    
os.mkdir(patch_dir)
n_subset = len(glob.glob(f'{img_dir}/*subset[0-9]'))

# Iterate over all the subsets
for i in range(n_subset):
    
    # Subset name
    subset = f'subset{i}'
    
    # Set the image and mask sub directories
    img_subset_dir = f'{img_dir}/{subset}/{subset}'
    msk_subset_dir = f'{msk_dir}/{subset}/{subset}'
    
    patch_subset_dir = f'{patch_dir}/{subset}/{subset}'
    
    os.makedirs(patch_subset_dir)
        
    # Get all the image and mask files in the sub directories
    msk_files = sorted(glob.glob(f'{msk_subset_dir}/*.mhd'))
    img_files = sorted(glob.glob(f'{img_subset_dir}/*.mhd'))

    for img_file, msk_file in zip(img_files, msk_files):
        
        # Read the SimpleITK image
        img_array = read_mhd(img_file)
        msk_array = read_mhd(msk_file)
        
        # clip and normalize between 0-255
        img_array = clip_normalize(img_array, low=-1000, high=+1000)
        msk_array = clip_normalize(msk_array, low=-1000, high=+1000)
        
        # Get patches
        img_patches, msk_patches = get_patches(img_array, msk_array, patch_volume=(128,128,16))

        # Extract the file id from the file path
        file_id = img_file.split('/')[-1][:-4]
        
        # Save path
        save_path = f'{patch_subset_dir}/{file_id}'
        os.makedirs(save_path)
        
        save_patches(img_patches, msk_patches, save_path)
        
        
    # Print when a subset is complete
    print(f'{subset} complete')
    
    shutil.rmtree(img_subset_dir)
    shutil.rmtree(msk_subset_dir)
    
    break
os.listdir('/kaggle/working/patches/subset0/subset0')
img_patches = sorted(glob.glob('/kaggle/working/patches/subset0/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260/images/*mhd'))
msk_patches = sorted(glob.glob('/kaggle/working/patches/subset0/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260/masks/*mhd'))
len(img_patches)
def read_mhd(file_path):
    array = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(array)
    return array
c  = 300
x = read_mhd(img_patches[c])
y = read_mhd(msk_patches[c])
plt.imshow(x[:,:,8])
def read_mhd(file_path):
    array = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(array)
    array = np.transpose(array, (1,2,0))
    return array
def clip_normalize(image, low=-700, high=-600):
    img = np.clip(image, low, high)
    img = (img - low) / (high - low)
    img = (img*255).astype(np.uint8)
    return img
img_files = sorted(glob.glob('/kaggle/working/roi/image/**/*.mhd', recursive=True))
msk_files = sorted(glob.glob('/kaggle/working/roi/mask/**/*.mhd', recursive=True))
image_arr = read_mhd(img_files[1])
print(image_arr.shape)

mask_arr = read_mhd(msk_files[1])
print(mask_arr.shape)
# Get the channel with nodule for visualization

if len(np.unique(mask_arr))==1:
    c = mask_arr.shape[-1]//2
    print('No nodule')
else:
    c = int(np.median(np.where(mask_arr==1)[2]))
    print(c)
    
    
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes = axes.flatten()
axes[0].imshow(image_arr[:,:,c])
axes[0].axis('off')
axes[1].imshow(mask_arr[:,:,c])
axes[1].axis('off')
plt.tight_layout()
plt.show()
clipped_image_arr = clip_normalize(image_arr, -1000, 1000)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes = axes.flatten()
axes[0].imshow(clipped_image_arr[:,:,c])
axes[0].axis('off')
axes[1].imshow(mask_arr[:,:,c])
axes[1].axis('off')
plt.tight_layout()
plt.show()
