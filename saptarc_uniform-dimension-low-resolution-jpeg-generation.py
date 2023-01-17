%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dicom

import os

import scipy.ndimage

import matplotlib.pyplot as plt

import cv2



from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# Some constants 

INPUT_FOLDER = '../input/sample_images/'

patients = os.listdir(INPUT_FOLDER)

patients.sort()
# Load the scans in given folder path

def load_scan(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices
def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slice_number] = image[slice_number].astype(np.int16)

            

        image[slice_number] += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)
def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing

    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)



    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    

    return image, new_spacing
MIN_BOUND = -1000.0

MAX_BOUND = 400.0

    

def normalize(image):

    image = 255*(image.astype(np.float64) - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)

    image[image>255] = 255

    image[image<0] = 0

    return image.astype(np.uint8)
def threshold_image(image, thval=110):

    

    image[image>=thval]=255

    image[image<thval]=0

    

    return image
#for patient_no in range (len(patients)):

for patient_no in range (0,6):

    first_patient = load_scan(INPUT_FOLDER + patients[patient_no])

    first_patient_pixels = get_pixels_hu(first_patient)

    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])

    

    #folder= "patient%d/" %patient_no

    #newpath = OUTPUT_FOLDER+folder 

    #if not os.path.exists(newpath):

    #    os.makedirs(newpath)

        

    v,h,w=pix_resampled.shape

    base_im=np.zeros((512,512))

    sr=int((512-h)/2);

    sc=int((512-w)/2);

    

    #--------- take onle 250 slices around the middle one ------

    for i in range (int(v/2-125),int(v/2+125)): 

        pix=pix_resampled[i]

        pix_jpg=normalize(pix)

        

        #-----expand to 512x512 -----------------

        base_pix=base_im

        base_pix[sr:sr+h][:,sc:sc+w]=pix_jpg

        rescaled_pix=cv2.resize(base_pix,(250,250))

        th_pix=rescaled_pix.copy()

        th_pix=threshold_image(th_pix,111)

        

        #-------- show one of the slices ----------

        if i==125:

            plt.imshow(np.concatenate((rescaled_pix, th_pix), axis=1), cmap=plt.cm.gray)

            plt.show()

           

        

        #fname = "slice%d.jpg" %i  

        #cv2.imwrite(newpath+fname, rescaled_pix)



        

print('end of code')