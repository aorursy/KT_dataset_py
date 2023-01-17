# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install medpy
from medpy.io import load

image_data, image_header = load('../input/sample_dataset_for_testing/sample_dataset_for_testing/fullsampledata/subset4mask/1.3.6.1.4.1.14519.5.2.1.6279.6001.122763913896761494371822656720/123.tiff')
image_data
image_data.shape,image_data.dtype
height,width = image_data.shape
from medpy.io import header

header.get_pixel_spacing(image_header)
directory = '../input/sample_dataset_for_testing/sample_dataset_for_testing/fullsampledata/subset4mask/1.3.6.1.4.1.14519.5.2.1.6279.6001.122763913896761494371822656720/'
n_slices  = 310

height,width = 512,512
vol3d = np.zeros((n_slices,height,width))

ddd = [os.path.join(directory, fname)

             for fname in os.listdir(directory) if fname.endswith('.tiff')]

for i,j in enumerate(ddd):

    vol3d[i],_ = load(str(j))
vol3d.shape
import matplotlib.pyplot as plt

%matplotlib inline

plt.imshow(vol3d[-1,:,:])

plt.colorbar()

plt.plot()
def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):

    fig,ax = plt.subplots(rows,cols,figsize=[12,12])

    for i in range(rows*cols):

        ind = start_with + i*show_every

        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)

        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')

        ax[int(i/rows),int(i % rows)].axis('off')

    plt.show()



sample_stack(vol3d)
plt.hist(vol3d.flatten(), bins=50, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()