%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dicom

import os

import scipy.ndimage

import matplotlib.pyplot as plt

import tensorflow as tf



from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# Input data files are available in the "../input/" directory.



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
  # Useful Methods

# Credits - https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial/notebook



# Load the scans in given folder path

def load_scan(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



#Method for reading HUs in the images http://gdcm.sourceforge.net/wiki/index.php/Hounsfield_Unit

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





# Plot DICOM image histogram

def plot_histo(pixels):

    plt.hist(pixels.flatten(), bins=80, color='c')

    plt.xlabel("Hounsfield Units (HU)")

    plt.ylabel("Frequency")

    plt.show()



# Plot image histogram

def plot_im(pixels,pixelcount):

    # Show some slice in the middle

    plt.imshow(pixels[pixelcount], cmap=plt.cm.gray)

    plt.show()



# Resample the image

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



# Method for plotting a 3D image

def plot_3d(image, threshold=-300):

    

    # Position the scan upright, 

    # so the head of the patient would be at the top facing the camera

    p = image.transpose(2,1,0)

    

    verts, faces,r1,r2 = measure.marching_cubes_lewiner(p, threshold)



    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')



    # Fancy indexing: `verts[faces]` to generate a collection of triangles

    mesh = Poly3DCollection(verts[faces], alpha=0.70)

    face_color = [0.45, 0.45, 0.75]

    mesh.set_facecolor(face_color)

    ax.add_collection3d(mesh)



    ax.set_xlim(0, p.shape[0])

    ax.set_ylim(0, p.shape[1])

    ax.set_zlim(0, p.shape[2])



    plt.show()



# Plot a dicom image    

def plt_dicom(image):

    

    pixels=get_pixels_hu(image)

    print("Histogram")

    #Plot histogram

    plot_histo(pixels)



    print("Image")

    #Plot some pixels from middle

    plot_im(pixels,0)



    pix_resampled,spacing=resample(pixels,image, [1,1,1])



    print("3D Image")

    plot_3d(pix_resampled,400)    
# Get list of all files in sample folder

INPUT_PATH="../input/sample_images/"

patients = os.listdir(INPUT_PATH)

patients.sort()

print(patients)
# Let us visualize one of the data files

path_first=INPUT_PATH + patients[0]

# Get the slices

image_slices=load_scan(path_first) 



print("Slices Count: ", len(image_slices))



# Get pixels first

plt_dicom(image_slices)
