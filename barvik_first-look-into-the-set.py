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
df = pd.read_csv ('../input/road-weather-information-stations.csv')

print(df.columns)
df.head(5)
df.tail(5)
#Get ranges of data and understand the wrong data, use 1% and 99% for wrong values determination
df[['RoadSurfaceTemperature','AirTemperature']].describe(percentiles=[.01, 0.25, 0.5, 0.75, 0.99])
#total count
count = df['RoadSurfaceTemperature'].count()
#counts of wrong values
bad_surface_count = (df[df['RoadSurfaceTemperature'] < 24])['RoadSurfaceTemperature'].count()
#for Air Temperature
bad_air_count = (df[(df['AirTemperature'] < 30) | (df['AirTemperature']>100)])['AirTemperature'].count()
print ('Total count is %d, bad surface count is %d, bad air count is %d'% (count, bad_surface_count, bad_air_count))
print ('or in %% bad surface temp. measurements is %.4f%%, bad air is %.4f%%' % (float(bad_surface_count)/float(count)*100.0, float(bad_air_count)/float(count)*100))
#get hist for correct values
mask = df['RoadSurfaceTemperature'] > 24 
mask &= df['AirTemperature'] > 30
mask &= df['AirTemperature'] < 100
df[mask].hist(column=['RoadSurfaceTemperature', 'AirTemperature'], sharex=True, sharey=True)