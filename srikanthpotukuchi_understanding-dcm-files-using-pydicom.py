# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pydicom
# methods in pydicom?

dir(pydicom)

# I see a read_file and dcmread
# Version check

pydicom.__version__
# Read dcm file

ds = pydicom.read_file("/kaggle/input/osic-pulmonary-fibrosis-progression/test/ID00421637202311550012437/32.dcm")
# Explore the dcm file

ds
# Let's try dcmread

ds2 = pydicom.dcmread("/kaggle/input/osic-pulmonary-fibrosis-progression/test/ID00421637202311550012437/32.dcm")
# Is ds2 same as ds?

ds2

# Looks like it's the same. I think read_file will be removed in future. See this - https://github.com/pydicom/pydicom/issues/475

# Below we can see the information for patient id - 'ID00421637202311550012437' and the CT scan of chest and details stored in pixel data
import matplotlib.pyplot as plt
# Credits: https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_read_dicom.html#sphx-glr-auto-examples-input-output-plot-read-dicom-py

if 'PixelData' in ds2:

    rows = int(ds2.Rows)

    cols = int(ds2.Columns)

    print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

        rows=rows, cols=cols, size=len(ds2.PixelData)))

    if 'PixelSpacing' in ds2:

        print("Pixel spacing....:", ds2.PixelSpacing)

        

# Pixel data can be used to plot the CT scan       
help(plt.imshow) # read more about imshow here: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html
# plot the image using matplotlib

plt.imshow(ds2.pixel_array, cmap=plt.cm.bone) 

plt.show()

# We see the chest CT scan of the patient.
# Let's try dcmread another file

ds3 = pydicom.dcmread("/kaggle/input/osic-pulmonary-fibrosis-progression/test/ID00421637202311550012437/18.dcm")
if 'PixelData' in ds3:

    rows = int(ds3.Rows)

    cols = int(ds3.Columns)

    print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

        rows=rows, cols=cols, size=len(ds3.PixelData)))

    if 'PixelSpacing' in ds3:

        print("Pixel spacing....:", ds3.PixelSpacing)
plt.imshow(ds3.pixel_array, cmap=plt.cm.bone)

plt.show()
import os

def read_all_dcm(PatientID):

    Path = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/" + PatientID + "/"

    path, dirs, files = next(os.walk(Path))

    file_count = len(files)

    PathList = [0]*file_count

    for i in range(len(PathList)):   

        try:

            PathList[i+1] = Path + str(i+1) + ".dcm"

            plt.imshow(pydicom.dcmread(PathList[i+1]).pixel_array, cmap=plt.cm.bone)

            plt.show()

            print(PathList[i+1])

        except:

            pass

        

    return
read_all_dcm("ID00421637202311550012437") # Below you can see all CT scans of the patient selected.