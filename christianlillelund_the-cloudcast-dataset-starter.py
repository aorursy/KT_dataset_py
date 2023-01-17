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

        break



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Print first 10 coordinates from imageset 2018M10



geo = np.load('/kaggle/input/the-cloudcast-dataset/2018M10/GEO.npz')

print (geo['lats'][:10], '\n')

print (geo['lons'][:10], '\n')



# Print first 10 timestamps from imageset 2018M10

timestamps = np.load('/kaggle/input/the-cloudcast-dataset/2018M10/TIMESTAMPS.npy')

print(timestamps[:10])
# Show first 10 image labels from imageset 2018M10



images = [np.load(f'/kaggle/input/the-cloudcast-dataset/2018M10/{i}.npy') for i in range(0,10)]

for img in images:

    print(img)
# Show a few of the satellite images



import matplotlib.pyplot as plt

plt.figure()

plt.imshow(images[0], cmap='gray')
plt.figure()

plt.imshow(images[1], cmap='gray')
plt.figure()

plt.imshow(images[2], cmap='gray')
plt.figure()

plt.imshow(images[3], cmap='gray')