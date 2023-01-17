import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



import pydicom as dcm

from pydicom.pixel_data_handlers.util import apply_modality_lut
import matplotlib.pyplot as plt

from skimage.segmentation import clear_border

from skimage.measure import label, regionprops

from skimage.morphology import disk, dilation, binary_erosion, binary_closing

from skimage.filters import roberts, sobel

import cv2

from scipy import ndimage as ndi



def get_segmented_lungs(im2, plot=False):

    im = im2.copy()

    # Step 1: Convert into a binary image.

    binary = im < -400

    

    if plot:

        plt.imshow(binary)

        plt.show()

        

    # Step 2: Remove the blobs connected to the border of the image.

    cleared = clear_border(binary)

    

    if plot:

        plt.imshow(cleared)

        plt.show()    

        

    # Step 3: Label the image.

    label_image = label(cleared)

    

    if plot:

        plt.imshow(label_image)

        plt.show()    

        

    # Step 4: Keep the labels with 2 largest areas.

    areas = [r.area for r in regionprops(label_image)]

    areas.sort()

    if len(areas) > 0:

        for region in regionprops(label_image):

            if region.area < areas[0]:

                for coordinates in region.coords:

                       label_image[coordinates[0], coordinates[1]] = 0

    binary = label_image > 0

    

    if plot:

        plt.imshow(binary)

        plt.show()  

        

    # Step 5: Erosion operation with a disk of radius 2. This operation is seperate the lung nodules attached to the blood vessels.

    selem = disk(2)

    binary = binary_erosion(binary, selem)

    

    if plot:

        plt.imshow(binary)

        plt.show()  

        

    # Step 6: Closure operation with a disk of radius 10. This operation is to keep nodules attached to the lung wall.

    selem = disk(10) # CHANGE BACK TO 10

    binary = binary_closing(binary, selem)

    

    if plot:

        plt.imshow(binary)

        plt.show() 

        

    # Step 7: Fill in the small holes inside the binary mask of lungs.

    edges = roberts(binary)

    

    if plot:

        plt.imshow(edges)

        plt.show() 

        

    binary = ndi.binary_fill_holes(edges)

    

    if plot:

        plt.imshow(binary)

        plt.show() 

        

    # Step 8: Superimpose the binary mask on the input image.

    selem = disk(4)

    binary = dilation(binary, selem)

    get_high_vals = binary == 0

    im[get_high_vals] = -2000

    

    if plot:

        plt.imshow(im)

        plt.show()

        

    return im, binary
train = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")
train.head()
data_path = "../input/rsna-str-pulmonary-embolism-detection/train/"



studyID, SeriesID, SOPID = train.loc[50,['StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID']].values
dicom = data_path+studyID+"/"+SeriesID+"/"+SOPID+".dcm"

img = dcm.dcmread(dicom)

img_data = img.pixel_array # Read the pixel value

hu = apply_modality_lut(img_data, img) # Transform to HU value

lung_seg, _ = get_segmented_lungs(hu)
plt.figure(figsize = (10,10))

plt.subplot(121)

plt.imshow(hu)

plt.subplot(122)

plt.imshow(lung_seg)

plt.show()
one_series_path = data_path+studyID+"/"+SeriesID+"/"

one_series = []

one_series_seg = []



for i in os.listdir(one_series_path):

    dicom_path = one_series_path+"/"+i

    img = dcm.dcmread(dicom_path)

    img_data = img.pixel_array

    hu = apply_modality_lut(img_data, img)

    img_seg, _ = get_segmented_lungs(hu)

    length = int(img.InstanceNumber)

    one_series.append((length, img_data))

    one_series_seg.append((length, img_seg))



one_series.sort()

one_series_seg.sort()

one_series_seg = [s[1] for s in one_series_seg]

one_series = [s[1] for s in one_series]
from matplotlib import animation, rc

rc('animation', html='jshtml')



def animate(ims,ims_seg):

    fig , (ax1, ax2) = plt.subplots(1,2,figsize=(15,8))

    ax1.axis('off')

    ax2.axis('off')

    im = ax1.imshow(ims[0])

    im2 = ax2.imshow(ims_seg[0])



    def animate_func(i):

        im.set_data(ims[i])

        im2.set_data(ims_seg[i])

        return [im,im2]



    anim = animation.FuncAnimation(fig, animate_func, frames = len(ims), interval = 1000//24)

    

    return anim

movie = animate(one_series,one_series_seg)
movie