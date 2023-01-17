# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))



import numpy as np
import h5py 
f = h5py.File('../input/mytestfile0.hdf5', 'r')
print(f.keys())
m=f['bigFFT'][...]
f.close()

# Any results you write to the current directory are saved as output.
def binning(input,binSize=10):
    binNumbers=input.shape[1] // binSize
    output=np.zeros((input.shape[0], binNumbers))
    for i in range(binNumbers):
        output[:,i]=np.mean(input[:,binSize*i:binSize*(i+1)],axis=1)

    return output
import matplotlib.pyplot as plt
mb=binning(m,10)
plt.hist(mb[1,:],40)
plt.show()
