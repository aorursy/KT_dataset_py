# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/crimes-in-boston/crime.csv', encoding='latin-1')
offence_groups = set(df['OFFENSE_CODE_GROUP'].tolist())

offence_groups = list(offence_groups)

print(offence_groups)
Latitudes = {}

Longitudes = {}



for offence in offence_groups:

    localLatitude = []

    localLongitude = []

    for location in df[df['OFFENSE_CODE_GROUP'] == offence]['Location']:

        Coordinates = location[1:-1].split(', ')

        latitude, longitude = float(Coordinates[0]),float(Coordinates[1])



        if latitude > 20:

            localLatitude.append(latitude)

        if longitude < -40:    

            localLongitude.append(longitude)

    

    Latitudes[offence] = localLatitude

    Longitudes[offence] = localLongitude
%matplotlib notebook

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation



fig, ax = plt.subplots(figsize=(8,6))



scat = ax.scatter(Longitudes[offence_groups[0]], Latitudes[offence_groups[0]], s=1, color='r')

 

def animate(i):

    x_i = Longitudes[offence_groups[i]]

    y_i = Latitudes[offence_groups[i]]

    scat.set_offsets(np.c_[x_i, y_i])

    

anim = FuncAnimation(

    fig, animate, interval=500, frames=len(offence_groups)-1)

plt.show()



anim.save('filename.gif', writer='imagemagick')