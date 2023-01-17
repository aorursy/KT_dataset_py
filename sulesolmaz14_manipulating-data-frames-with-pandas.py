# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir('../input'))

# Any results you write to the current directory are saved as output.
#read data
data = pd.read_csv('../input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-20-population-contracted.csv')
#data = data.set_index('#')
data.head()
#indexing using square brakets
data["total_hospital_beds"][0]
#or
data.total_hospital_beds[0]
#using loc accessor
data.loc[1,["total_hospital_beds"]]
#selecting only some column
data[["total_hospital_beds", "total_icu_beds"]]
#Difference between selecting columns: series, dataframe
print(type(data["total_hospital_beds"]))
print(type(data[["total_hospital_beds"]]))
### Slicing and indexing series
data.loc[0:10, "total_hospital_beds": "total_icu_beds"]
#Reverse slicing
data.loc[10:0:-1, "total_hospital_beds": "total_icu_beds"]
#From something to end
data.loc[1:10, "total_hospital_beds":]
### Creating boolean series
boolean = data.total_hospital_beds>6894.0
data[boolean]
#Combining filters
first_filter = data.hospital_bed_occupancy_rate>0.62
second_filter = data.icu_bed_occupancy_rate>0.55
data[first_filter & second_filter]
#Filtering column based others
data.hospital_bed_occupancy_rate[data.icu_bed_occupancy_rate<0.62]
#Plain python function
def div(n):
    return n/2
data.total_hospital_beds.apply(div)
#We can use lambda function
data.total_hospital_beds.apply(lambda n : n/2)
#Defining column using other columns
data["Total Bed"] = data.total_hospital_beds + data.total_icu_beds
data.head()