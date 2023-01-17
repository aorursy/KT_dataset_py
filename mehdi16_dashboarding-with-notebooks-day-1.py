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
data = pd.read_csv("../input/red-light-camera-violations.csv")
locationsdata = pd.read_csv("../input/red-light-camera-locations.csv")

data.head()
locationsdata.head()
import matplotlib.pyplot as plt

#show the top 10 intersections where violations occur
# this allows to focus on those for on site analysis

plt.rcParams['figure.figsize'] = [15, 5]

p_camerapivot = data.pivot_table(index="INTERSECTION", values ="VIOLATIONS").nlargest(10, "VIOLATIONS")


p_camerapivot.plot.bar()

plt.show()
locationsdata['INTERSECTION']=locationsdata['INTERSECTION'].fillna('')
locationsdata['INTERSECTION']=locationsdata['INTERSECTION'].str.upper()


locationsdata.query('INTERSECTION in @p_camerapivot.index')
#no match because of different spelling
# credits to Beza https://www.kaggle.com/bezget/dashboarding-with-notebooks-day-1/notebook
data['VIOLATIONDATEDAY'] = pd.to_datetime(data['VIOLATION DATE'])
# Commented sorting and indexing because it would cause errors later on
#data.sort_values(by='VIOLATIONDATEDAY',inplace=True)
#data.set_index('VIOLATIONDATEDAY',inplace=True)
data.head()
import datetime as dt

data['MONTH']=pd.DatetimeIndex(data['VIOLATIONDATEDAY']).month 

p_monthpivot = data.pivot_table(index="MONTH", values ="VIOLATIONS")
data.sort_values(by='MONTH',inplace=True)
#data.set_index('VIOLATIONDATEDAY',inplace=True)

p_monthpivot.plot.bar()


 


data['DOW']=pd.DatetimeIndex(data['VIOLATIONDATEDAY']).dayofweek 


p_dayofweekpivot = data.pivot_table(index="DOW", values ="VIOLATIONS")
data.sort_values(by='DOW',inplace=True)
data.set_index('DOW',inplace=True)

p_dayofweekpivot.plot.bar()