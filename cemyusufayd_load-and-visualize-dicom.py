!conda install -c conda-forge pydicom gdcm -y
import numpy as np # linear algebra
import os
import pydicom
import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
filepath = "/kaggle/input/covid19-ct-scans/Case_002/coronacases_002_171.dcm"
dcmfile = pydicom.dcmread(filepath)
dcmfile
dcm_numpy = dcmfile.pixel_array
dcm_numpy.shape
plt.imshow(dcm_numpy, cmap=plt.cm.bone)
