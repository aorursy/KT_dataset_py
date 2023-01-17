### ########################################### 

### Simple Dicom Viewer Code 

### Author : Taposh Dutta Roy 

### ########################################### 



# The imports 

import cv2

import numpy as np

import matplotlib.pyplot as plt

import dicom as pdicom

import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

########################

# List of all dicom images

########################

dicom_path = "../input/sample_images/"

lstFilesDCM = []  # create an empty list

for dirName, subdirList, fileList in os.walk(dicom_path):

    for filename in fileList:

        if ".dcm" in filename.lower():  # check whether the file's DICOM

            lstFilesDCM.append(os.path.join(dirName,filename))

            #print(lstFilesDCM)

#print(len(lstFilesDCM))



####################

#Fun with Dicom

####################



# Get reference file

RefDs = pdicom.read_file(lstFilesDCM[0])



#See the ref file

print(RefDs)







# Load dimensions based on the number of rows, columns, and slices (along the Z axis)

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

# Load spacing values (in mm)

ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]))

x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])

y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])





ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)



# loop through all the DICOM files

for filenameDCM in lstFilesDCM:

    # read the file

    ds = pdicom.read_file(filenameDCM)

    # store the raw image data

    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  

#Plot the figure 

plt.figure(dpi=1200)

plt.axes().set_aspect('equal', 'datalim')

plt.set_cmap(plt.gray())

plt.pcolormesh(x, y, np.flipud(ArrayDicom[:, :, 157]))



# Show the Dicom Image

plt.show()