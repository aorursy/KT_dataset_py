# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import subprocess

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

# check the files
# print (subprocess.check_output('ls ../input/'))
stations = pd.read_csv('../input/station.csv')
# statuses = pd.read_csv('../input/status.csv')
trips = pd.read_csv('../input/trip.csv')
weather = pd.read_csv('../input/weather.csv')
# Some interesting tid-bits of info
cities_with_bike_share = stations['city'].unique()
print(cities_with_bike_share)

