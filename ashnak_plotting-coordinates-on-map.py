# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Path of the file to read
data_file_path = '../input/nville-points/NorthvilleRoute.txt'
df = pd.read_csv(data_file_path)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df.head()

#BBox = ((df.longitude.min(),   df.longitude.max(),      
#         df.latitude.min(), df.latitude.max())
BBox = (-83.5661,-83.4683, 42.3910, 42.4359)
ruh_m = plt.imread('../input/nville-map2/NorthvilleMap.png')
fig, ax = plt.subplots(figsize = (24,21))
ax.scatter(df.Longitude, df.Latitude, zorder=1, alpha= 0.5, c='r', s=20)
ax.set_title('Northville Route')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')