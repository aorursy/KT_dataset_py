%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dicom

import os

import scipy.ndimage

import matplotlib.pyplot as plt



from skimage import measure

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# Some constants 

INPUT_FOLDER = '../input/sample_images/'

patients = os.listdir(INPUT_FOLDER)

patients.sort()
# Load the scans in given folder path

def load_scan(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    

    slice_thickness = slices[0].SliceLocation - slices[1].SliceLocation

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices
def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans])

    

    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

      

    intercept = scans[0].RescaleIntercept

    image += int(intercept)

    

    return np.array(image)
first_patient = load_scan(INPUT_FOLDER + patients[0])

first_patient_pixels = get_pixels_hu(first_patient)

plt.hist(first_patient_pixels.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()



# Show some slice in the middle

plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)

plt.show()
def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing

    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))

    spacing = np.array(list(spacing))



    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    

    return image, new_spacing
pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])

print("Shape before resampling\t", first_patient_pixels.shape)

print("Shape after resampling\t", pix_resampled.shape)
def plot_3d(image, threshold=-300):

    

    # Position the scan upright, 

    # so the head of the patient would be at the top facing the camera

    p = image.transpose(2,1,0)

    p = p[:,:,::-1]

    

    verts, faces = measure.marching_cubes(p, threshold)



    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')



    # Fancy indexing: `verts[faces]` to generate a collection of triangles

    mesh = Poly3DCollection(verts[faces], alpha=0.1)

    face_color = [0.5, 0.5, 1]

    mesh.set_facecolor(face_color)

    ax.add_collection3d(mesh)



    ax.set_xlabel("x-axis")

    ax.set_ylabel("y-axis")

    ax.set_zlabel("z-axis")



    ax.set_xlim(0, p.shape[0])  # a = 6 (times two for 2nd ellipsoid)

    ax.set_ylim(0, p.shape[1])  # b = 10

    ax.set_zlim(0, p.shape[2])  # c = 16



    plt.show()
plot_3d(pix_resampled, 400)
MIN_BOUND = -1000.0

MAX_BOUND = 400.0

    

def normalize(image):

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)

    image[image>1] = 1.

    image[image<0] = 0.

    return image
PIXEL_MEAN = 0.25



def zero_center(image):

    image = image - PIXEL_MEAN

    return image