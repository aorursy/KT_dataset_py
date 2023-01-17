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
cities = pd.read_csv ("Dekstop/cities_r2.csv")
cities.head ()
cities.info ()
cities.describe ()
print (cities.describe(include=['O']))
fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['name_of_city'].count().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('No of cities taken for analysis', fontsize = 20)

plt.show ()
import numpy as np

from scipy.interpolate import spline

from numpy import array

import matplotlib as mpl



# for plots

import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.dates import date2num

from mpl_toolkits.basemap import Basemap



# for date and time processing

import datetime



# for statistical graphs

import seaborn as sns
import pandas as pd

cities = pd.read_csv("../input/cities_r2.csv")