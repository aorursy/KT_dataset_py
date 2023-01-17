# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper 
from bq_helper import BigQueryHelper
import matplotlib.pyplot as plt
import pandas as pd 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

usfs = BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="usfs_fia")






df_facilityuse = pd.read_csv( '../input/usfs-fia/state-park-annual-attendance-figures-by-facility-beginning-2003.csv')

## How has attendance changed over the years?
groupedYear  = df_facilityuse.groupby( [ 'Year' ]).agg({'Attendance': 'sum'})
groupedYear.plot(legend=False)




# Lets find the most attended facilities
mostAttendedFacilities = df_facilityuse.groupby( [  'Facility' ]).agg({'Attendance': 'sum'}).sort_values('Attendance')
n = 20
topN = mostAttendedFacilities.nlargest(n, 'Attendance')


# Now lets plot only these and how their attendance changed over the years

df_topN = df_facilityuse[ df_facilityuse['Facility'].isin(topN.index) ]
groupedYear  = df_topN.groupby( [ 'Year', 'Facility' ]).agg({'Attendance': 'sum'})
groupedYear = groupedYear.unstack()
groupedYear.plot(legend=True, figsize=(20,8))
## Now lets see how the forest changes around these areas..

## We have to fix some names...
names = topN.index.tolist()
names = [name.replace('St Pk', '') for name in names]
names = [name.replace('St Park', '') for name in names]
names = [name.replace('State Park', '') for name in names]
names = [name.replace('- Long Island', '') for name in names]
names = [name.replace('Saratoga Springs', 'Saratoga Spa') for name in names]
names = [name.replace('Niagara Reservation', 'Niagara Falls') for name in names]
names = [name.replace('Sunken Meadow', 'Alfred E. Smith/Sunken Meadow') for name in names]

names = [name.strip() for name in names]

#Import facilities data
df_facilities = pd.read_csv( '../input/nys-state-park-annual-attendance-figures/State_Park_Facility_Points.csv')

for i, row in enumerate(names):
    facility = df_facilities[ df_facilities['Name'].str.contains(row)]
    
    name = getattr(facility, "Name").values[0]
    
    print(name )

    lat = getattr(facility, "Latitude").values[0]
    lon = getattr(facility, "Longitude").values[0]
    
    ## Lets get some information about the trees surrounding this park. 
    query1 = """
    SELECT
        total_height,
        current_diameter,
        measurement_year,
        plot_county_code,
        latitude,
        longitude,
        elevation,
        growth_trees_per_acre_unadjusted as growth,
        trees_per_acre_unadjusted as trees
    FROM (select *, \
       ((ACOS(SIN({latitude} *  ACOS(-1)/ 180) * \
              SIN(latitude  *  ACOS(-1) / 180) + \
              COS({latitude} *  ACOS(-1) / 180) * \
              COS(latitude   *  ACOS(-1) / 180) * \
              COS(({longitude} - longitude) *  ACOS(-1) / 180)) * \
         180 /  ACOS(-1) ) * \
        60 * 1.1508) as distance_miles \
    FROM
        `bigquery-public-data.usfs_fia.plot_tree`)
    WHERE
        distance_miles < 10
    ;        """.format( latitude=lat, longitude=lon)

    
    response1 = usfs.query_to_pandas_safe(query1, max_gb_scanned=10)
        
    df = response1.groupby([ 'measurement_year']).agg({'current_diameter': 'mean', 'total_height': 'mean', 'growth': 'mean', 'trees': 'mean'  })
     
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
           
    df.plot( title=name , ax=axes[0] )
    
    df_attendance = df_facilityuse[ df_facilityuse['Facility'].str.contains(topN.index[i]) ]
    groupedYear  = df_attendance.groupby( [ 'Year'] ).agg({'Attendance': 'sum'})
    groupedYear.plot(  ax=axes[1])
    

    