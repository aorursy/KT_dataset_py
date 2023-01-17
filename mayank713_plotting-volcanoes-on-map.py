# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/volcanic-eruptions/database.csv')

world_map = cv2.imread('/kaggle/input/world-map/world_map.png')
plt.imshow(world_map)
data.isnull().sum()
data.info()
elevation = data['Elevation (Meters)'].mean()

norm = plt.Normalize(elevation)

sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)

sm.set_array([])

plt.figure(figsize=(15,8))

plt.imshow(world_map, zorder=0, extent=[-200, 200, -100, 86])

ax = plt.gca()

sns.scatterplot(x='Longitude', y='Latitude', data=data, hue='Elevation (Meters)', palette='Reds')

ax.get_legend().remove()

ax.figure.colorbar(sm)

