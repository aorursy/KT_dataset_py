# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
level = pd.read_csv("../input/chennai_reservoir_levels.csv")

rain = pd.read_csv("../input/chennai_reservoir_rainfall.csv")

level.head()
level.tail()
level.shape
level.describe()
rain.head()

rain.tail()
rain.describe()
rain.shape
#convert columns to timestamp

level['Date'] = pd.to_datetime(level['Date'])

rain['Date'] = pd.to_datetime(rain['Date'])

#level['Date'].dt.year.head()
#level.info()
mean_level = level.groupby(level['Date'].dt.year).agg(np.mean).reset_index()

mean_level
mean_rain = rain.groupby(rain['Date'].dt.year).agg(np.mean).reset_index()

mean_rain
reservoirs = ['POONDI','CHOLAVARAM','REDHILLS','CHEMBARAMBAKKAM']

plt.figure(figsize=(10,6))

for reservoir in reservoirs:

    plt.plot(mean_level['Date'],mean_level[reservoir])



#1. Redhills - green

#2. Chembarambakkam - red

#3. Poondi - blue

#4. Cholavaram - orange
plt.figure(figsize=(10,6))

for reservoir in reservoirs:

    plt.plot(mean_rain['Date'],mean_rain[reservoir])



#1. Chembarambakkam - red

#2. Cholavaram -orange

#3. Redhills - green

#4. Poondi - blue
#Few insights

# 1 Although cholavaram got the same amount of rainfall as the other areas, the water level at that reservoir has been very low relative to other reservoirs. It might be due to poor infra or 

#   the capacity of that reservoir is just less. Probably a lot of water just gets wasted from that reservoir and goes to sea.

# 2 The situation in 2019 seems to very bad than 2004 since the end of the rainfall plot just dips down and trend might probably just continue to dip down and get even worse!