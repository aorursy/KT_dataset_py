import numpy as np, pandas as pd, os

import matplotlib.pyplot as plt
!cp ../input/gdcm-conda-install/gdcm.tar .

!tar -xvzf gdcm.tar

!conda install --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2

print("done")
mypath = '../input/rsna-str-pulmonary-embolism-detection/train'



import pydicom



bad_file = '0af5610bf683/6ee796097155/09c3501f2449.dcm'

dataset = pydicom.read_file(mypath+'/'+ bad_file)

img = dataset.pixel_array

plt.imshow(img,'gray')

plt.show()



#reads the entire directory tree if you want to do more testing (assumes a single SeriesInstanceUID)

#

#dirnames = os.listdir(mypath)

#for dire in dirnames:

#    seriesnames = os.listdir(mypath+'/'+dire)

#    filenames = os.listdir(mypath+'/'+dire+'/'+seriesnames[0])

#    for filename in filenames:

#        dataset = pydicom.read_file(mypath+'/'+ dire +'/'+seriesnames[0] + '/' +filename)

#        print(dire, filename)

#        img = dataset.pixel_array


