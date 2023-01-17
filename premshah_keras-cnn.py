# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import h5py
h5f = h5py.File('../input/LowRes_13434_overlapping_pairs.h5','r')

pairs = h5f['dataset_1'][:]

h5f.close()
from matplotlib import pyplot as plt



grey = pairs[220,:,:,0]

mask = pairs[220,:,:,1]

#%matplotlib inline

plt.subplot(121)

plt.imshow(grey)

plt.title('max='+str(grey.max()))

plt.subplot(122)

plt.imshow(mask)
pairs.size
grey.size
images = pairs[:,:,:,0]
image_size= images.shape[1:]

output= pairs[:,:,:,1]
output[1]
plt.imshow(output[1])