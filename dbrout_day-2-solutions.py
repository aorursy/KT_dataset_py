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
import matplotlib.pyplot as plt

#plt.scatter(xdat,ydat)

#plt.plot(xmatter,ymatter)

#plt.plot(xaccel,yaccel)

from scipy.interpolate import interp1d

f = interp1d(xaccel,yaccel) # default is linear

plt.scatter(xdat,ydat-f(xdat))

plt.plot(xmatter,ymatter-f(xmatter))



plt.axhline(0)
wherebig = (population > 200000)



fig = plt.figure(figsize=(8, 8))

centerlat = np.mean(lat)

centerlon = np.mean(lon)

m = Basemap(projection='lcc', #resolution='h', 

            lat_0=centerlat, lon_0=centerlon,

            width=1E6, height=1.2E6)

m.shadedrelief()

m.drawcoastlines(color='gray')

m.drawstates(color='gray')

m.scatter(lon[wherebig], lat[wherebig], latlon=True, 

          s=(population[wherebig]-200000)**.5, alpha=0.5)



# make legend with dummy points

for a in [500000, 1000000, 5000000]:

    plt.scatter([], [], c='k', alpha=0.5, s=(a-200000)**.5,

                label= '%d Thousand'%(a/1000))

plt.legend(frameon=False,labelspacing=3,

           loc='lower left')