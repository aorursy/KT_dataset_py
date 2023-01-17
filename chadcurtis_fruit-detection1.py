# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
!ls /kaggle/input/fruit-detection/images
im = plt.imread('/kaggle/input/fruit-detection/images/fruit150.png')

plt.imshow(im)

plt.axis('off')
im.shape
ims = np.zeros((600, 600, 4, 212))

# for i in range(213):

#     try:

#         ims[:,:,:,i] = plt.imread('/kaggle/input/fruit-detection/images/fruit{}.png'.format(i))

#     except:

#         im = plt.imread('/kaggle/input/fruit-detection/images/fruit{}.png'.format(i))

#         im = np.swapaxes(im,0,1)

#         ims[:,:,:,i] = im



for i in range(213):

    try:

        im = plt.imread('/kaggle/input/fruit-detection/images/fruit{}.png'.format(i))

        size = im.shape

        x = size[0]

        y = size[1]

        channels = size[2]



        ims[0:x, 0:y, 0:channels, i] = im

    except:

        pass
ims[:,:,:,1].shape
plt.imshow(ims[:,:,:,100])
# Check for longest axis

# Flip axes so longest axis is x axis

# Rescale image so longest axis is equal to 400 pixels

# Check length of the shorter axis after resizing



# Option 1: stretch second axis so it's equal to 300

# Option 2: crop second axis so it's equal to 300