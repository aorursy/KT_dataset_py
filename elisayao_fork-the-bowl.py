import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import dicom

import os

import scipy.ndimage

import matplotlib.pyplot as plt

%matplotlib inline



from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
input_folder = '../input/sample_images/'

patients = os.listdir(input_folder)

patients.sort()
patients[0:10]
# Load the scans in given folder path

# Basically replicate the process of loading "../input/sample_images"



def load_scan(path):

    slices = [dicom.read_file(path + '/' + serial_number) for serial_number in os.listdir(path)]

    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))  

    # ImagePositionPatient is a piece of information about patient

    try:

        slice_thickness=np.abs(slices[0].ImagePositionPatient[2]-slices[1].ImagePositionPatient[2])

        # calculate thickness using the same position in coordinate

    except:

        slice_thickness=np.abs(slices[0].SliceLocation-slices[1].SliceLocation)

    

    for serial_number in slices:

        serial_number.SliceThickness = slice_thickness

    

    return slices
def get_pixels_hu(slices):

    image=np.stack([s.pixel_array for s in slices])

    image=image.astype(np.int16)  # convert to in16 data type

    

    image[image==-2000]=0

    # set the value of area out of scan to be 0

    

    # convert to Hounsfield units(HU)

    for slice_number in range(len(slices)):

        

        intercept=slices[slice_number].RescaleIntercept

        slope=slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number]=slop*image[slice_number].astype(np.float64)

            image[slice_number]=image[slice_number].astype(np.int16)

            

        image[slice_number] += np.int16(intercept)

        

    return np.array(image,dtype=np.int16)
# Look at one patient at a time



first_patient = load_scan(input_folder + patients[0])

first_patient_pixels = get_pixels_hu(first_patient)

plt.hist(first_patient_pixels.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()



# Show some slice in the middle

plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)

plt.show()
# resampling



def resample(image, scan, new_spacing=[1,1,1]):

    # determine current pixel spacing

    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    

    resize_factor = spacing / new_spacing  # percentage shrink

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    

    return image, new_spacing


# first patient as an example



pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])

print("Shape before resampling\t", first_patient_pixels.shape)

print("Shape after resampling\t", pix_resampled.shape)
# 3D Plotting

def plot_3d(image, threshold=-300):  # the threshold argument is to control what we want to plot

    # position the scan upright so the head of the patient would be at the top facing the camera

    p = image.transpose(2,1,0) # (z,y,x) w.r.t. original axes

    

    verts, faces = measure.marching_cubes(p, threshold)

    

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111, projection='3d')

    

    # Fancy indexing: 'verts[faces]' to generate a collection of triangles

    mesh = Poly3DCollection(verts[faces], alpha=0.70)

    face_color = [0.45, 0.45, 0.75]

    mesh.set_facecolor(face_color)

    ax.add_collection3d(mesh)

    

    ax.set_xlim(0, p.shape[0])

    ax.set_ylim(0, p.shape[1])

    ax.set_zlim(0, p.shape[2])

    

    plt.show()
# show the bones only, first patient

%time

plot_3d(pix_resampled, 400)
# lung segmentation

# connected component analysis



def largest_label_volume(im, bg=-1):

    vals, counts = np.unique(im, return_counts=True)  # unique values in an array

    

    counts = counts[vals != bg]

    vals = vals[vals != bg]

    

    if len(counts) > 0:

        return vals[np.argmax(counts)] # index of maximum in an array

    else:

        return None
def segment_lung_mask(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2

    # 0 is treated as background, which we do not want

    binary_image = np.array(image > -320, dtype=np.int8) + 1  # lung HU is -500. 

    # Why -320? Maybe empirical.

    labels = measure.label(binary_image) # label connected regions of an integer array

    

    # pick the pixel in the very corner to determine which label is air

    # improvement: pick multiple backgound labels from aroud the patient

    background_label = labels[0,0,0]  # pixel in the corner

    

    # fill the air around the person

    binary_image[background_label == labels] = 2

    

    # method of filling the lung structures

    if fill_lung_structures:  # if the second parameter is passed as "True"

        # for every slice we determine the largest solid structure

        for i, axial_slice in enumerate(binary_image):

            axial_slice = axial_slice - 1

            labeling = measure.label(axial_slice)

            l_max = largest_label_volume(labeling, bg=0)

            

            if l_max is not None: # this slice DOES contain some lung

                binary_image[i][labeling != l_max] = 1

                

    binary_image -= 1  # make the image actually binary

    binary_image = 1 - binary_image  # invert the labels so lung tissues are now 1

    

    # remove other air pockets inside body

    labels = measure.label(binary_image, background=0)

    l_max = largest_label_volume(labels, bg=0)

    

    if l_max is not None:  # there ARE air pockets

        binary_image[labels != l_max] = 0

        

    return binary_image
segmented_lungs = segment_lung_mask(pix_resampled, False)

segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
# plot 3D masked lung organ

plot_3d(segmented_lungs, 0)
# plot 3D masked lung, larger air pockets filled

plot_3d(segmented_lungs_fill, 0)
# the difference between the filled and non-filled

# structure of larger air pockets (bronchial?)

plot_3d(segmented_lungs_fill - segmented_lungs, 0)
# Normalization

# we are more interested in HU ranging from -1000 to 400, since anything higher than 400 are bones

min_bound = -1000.0

max_bound = 400.0



def normalize(image):

    image = (image - min_bound) / (max_bound - min_bound)  # normalization formula

    # scale the dataset into [0,1] range

    image[image>1] = 1.  # values that originally larger than 400

    image[image<0] = 0.  # values that originally smaller than -1000

    return image
# Zero centering

# it means scaling your data so that the mean value is 0



pixel_mean = 0.25  # empirical result from LUNA 16 competition



def zero_center(image):

    image = image - pixel_mean

    return image