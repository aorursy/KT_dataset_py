%matplotlib inline



import numpy as np, pandas as pd

import pydicom, imageio, os

import matplotlib.pyplot as plt

from IPython import display

from IPython.display import HTML

from glob import glob

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scipy.ndimage

from skimage import morphology

from skimage import measure

from skimage.transform import resize

from sklearn.cluster import KMeans

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.figure_factory as ff

from plotly.graph_objs import *

init_notebook_mode(connected=True) 
train_df = pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/train.csv')

test_df = pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/test.csv')





patient_id_list = os.listdir('../input/rsna-str-pulmonary-embolism-detection/train/')

print(f'Total number of patient(experiment): {len(patient_id_list)}\nFirst 5 patient IDs')

patient_id_list[:5]
index = patient_id_list.index('6897fa9de148')

patient_id = patient_id_list[index]

patient_folder = f'../input/rsna-str-pulmonary-embolism-detection/train/{patient_id}/'

patient_image_paths = glob(patient_folder + '/*/*.dcm')

[neg, pos] = train_df[train_df.StudyInstanceUID == patient_id].pe_present_on_image.value_counts().values





# Print out the first 5 file names to verify we're in the right folder.

print (f'Total of {len(patient_image_paths)} DICOM images.' )

print(f'The experiment {patient_id} has {pos} positive and {neg} negative examples for pe_present_on_image\nFirst 5 filenames:')



patient_image_paths[:5]
patient_image_paths[0][-16:-4]
def load_slice(paths):

    slices = [pydicom.read_file(path) for path in paths]

#     labels = [train_df[train_df.SOPInstanceUID == path[-16:-4]].pe_present_on_image.values for path in patient_image_paths]

#     labels = np.array(labels).squeeze()

    slices.sort(key = lambda x: int(x.InstanceNumber), reverse = False)

#     labels.sort(key = lambda x: int(x.InstanceNumber), reverse = False)

    

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)    

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



def transform_to_hu(slices):

    images = np.stack([file.pixel_array for file in slices])

    images = images.astype(np.int16)



    # convert ouside pixel-values to air:

    # I'm using <= -1000 to be sure that other defaults are captured as well

    images[images <= -1000] = 0

    

    # convert to HU

    for n in range(len(slices)):    

        intercept = slices[n].RescaleIntercept

        slope = slices[n].RescaleSlope

        if slope != 1:

            images[n] = slope * images[n].astype(np.float64)

            images[n] = images[n].astype(np.int16)      

        images[n] += np.int16(intercept)

    return np.array(images, dtype=np.int16)
stacked_dicoms = load_slice(patient_image_paths)

stacked_patient_pixels = transform_to_hu(stacked_dicoms)



def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):

    fig,ax = plt.subplots(rows,cols,figsize=[20,22])

    for i in range(rows*cols):

        ind = start_with + i*show_every

        ax[int(i/rows),int(i % rows)].set_title(f'slice {ind}')

        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')

        ax[int(i/rows),int(i % rows)].axis('off')

    plt.show()



     

sample_stack(stacked_patient_pixels, show_every = 3)
imageio.mimsave(f'./{patient_id}.gif', stacked_patient_pixels, duration=0.1)

display.Image(f'./{patient_id}.gif', format='png')
print(f'Slice Thickness: {stacked_dicoms[0].SliceThickness}')

print(f'Pixel Spacing (row, col): ({stacked_dicoms[0].PixelSpacing[0]}, {stacked_dicoms[0].PixelSpacing[1]})')
np.array([float(stacked_dicoms[0].SliceThickness), float(stacked_dicoms[0].PixelSpacing[0]), float(stacked_dicoms[0].PixelSpacing[0])])
def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing

#     spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))

    spacing = np.array([float(scan[0].SliceThickness), 

                        float(scan[0].PixelSpacing[0]), 

                        float(scan[0].PixelSpacing[0])])





    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    

    return image, new_spacing



print(f'Shape before resampling: {stacked_patient_pixels.shape}')

imgs_after_resamp, spacing = resample(stacked_patient_pixels, stacked_dicoms, [1,1,1])

print(f'Shape after resampling: {imgs_after_resamp.shape}')
def air_removal_mask(dilation):

    labels = measure.label(dilation)

    label_vals = np.unique(labels)

    if labels[0,0] == labels[-1, -1]:

        upper_cut = (labels==labels[0,0])

        mask = np.abs(upper_cut*1 -1) 

    else:

        upper_cut = (labels == labels[0,0])

        lower_cut = (labels == labels[-1,-1])

        mask = np.abs((upper_cut + lower_cut )*1 -1)      

    return mask
#Standardize the pixel values

def make_lungmask(img, display=False):

    row_size= img.shape[0]

    col_size = img.shape[1]

    

    mean = np.mean(img)

    std = np.std(img)

    img = img-mean

    img = img/std

    # Find the average pixel value near the lungs

    # to renormalize washed out images

    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 

    mean = np.mean(middle)  

    max = np.max(img)

    min = np.min(img)

    # To improve threshold finding, I'm moving the 

    # underflow and overflow on the pixel spectrum

    img[img==max]=mean

    img[img==min]=mean

    #

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)

    #

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))

    centers = sorted(kmeans.cluster_centers_.flatten())

    threshold = np.mean(centers)

    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image



    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  

    # We don't want to accidentally clip the lung.



    eroded = morphology.erosion(thresh_img,np.ones([3,3]))

    dilation = morphology.dilation(eroded,np.ones([8,8]))



    labels = measure.label(dilation) # Different labels are displayed in different colors

    label_vals = np.unique(labels)

    regions = measure.regionprops(labels)

    good_labels = []

    for prop in regions:

        B = prop.bbox

        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:

            good_labels.append(prop.label)

    mask = np.ndarray([row_size,col_size],dtype=np.int8)

    mask[:] = 0



    #

    #  After just the lungs are left, we do another large dilation

    #  in order to fill in and out the lung mask 

    #

    for N in good_labels:

        mask = mask + np.where(labels==N,1,0)

    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation [10,10]

    

    mask = dilation.astype('int16')*air_removal_mask(dilation)

    

    if (display):

        fig, ax = plt.subplots(3, 2, figsize=[12, 12])

        ax[0, 0].set_title("Original")

        ax[0, 0].imshow(img, cmap='gray')

        ax[0, 0].axis('off')

        

        ax[0, 1].set_title("Threshold")

        ax[0, 1].imshow(thresh_img, cmap='gray')

        ax[0, 1].axis('off')

        

        ax[1, 0].set_title("After Erosion and Dilation")

        ax[1, 0].imshow(dilation, cmap='gray')

        ax[1, 0].axis('off')

        

        ax[1, 1].set_title("Color Labels")

        ax[1, 1].imshow(labels)

        ax[1, 1].axis('off')

        

        ax[2, 0].set_title("Final Mask")

        ax[2, 0].imshow(mask, cmap='gray')

        ax[2, 0].axis('off')

        

        ax[2, 1].set_title("Apply Mask on Original")

        ax[2, 1].imshow(mask*img, cmap='gray')

        ax[2, 1].axis('off')

        

        plt.show()

    return mask*img

img = imgs_after_resamp[120]

output = make_lungmask(img, display=True)
from tqdm import tqdm

masked_lung = []



for img in tqdm(imgs_after_resamp):

    masked_lung.append(make_lungmask(img))



sample_stack(masked_lung, show_every=6)
imageio.mimsave(f'segmented{patient_id}.gif', masked_lung, duration=0.1)

display.Image(f'segmented{patient_id}.gif', format='png')
np.save(f'resampled_masked_lung_{patient_id}.npy', masked_lung)

np.save(f'stacked_slices_{patient_id}.npy', stacked_patient_pixels)