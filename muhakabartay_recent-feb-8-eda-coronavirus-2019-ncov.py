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
!ls ../input/novel-coronavirus-2019ncov/
# Reading the datasets

data_ncov_c= pd.read_csv("../input/novel-coronavirus-2019ncov/coronavirus_conf.csv")

data_ncov_r= pd.read_csv("../input/novel-coronavirus-2019ncov/coronavirus_reco.csv")

data_ncov_d= pd.read_csv("../input/novel-coronavirus-2019ncov/coronavirus_death.csv")

data_ncov_c.head(10)
# Confirmed countries and regions affected by virus

places = data_ncov_c['Country/Region'].unique().tolist()

print(places)



print("\nTotal countries and regions affected by virus: ",len(places))



## Hong Kong, Macau are separated from China
data_ncov_c.groupby(['Country/Region','Province/State']).sum()
from shapely.geometry import Point

import geopandas as gpd

from geopandas import GeoDataFrame



geometry_c = [Point(xy) for xy in zip(data_ncov_c['Long'].astype('float'), data_ncov_c['Lat'].astype('float'))]

geometry_r = [Point(xy) for xy in zip(data_ncov_r['Long'].astype('float'), data_ncov_r['Lat'].astype('float'))]

geometry_d = [Point(xy) for xy in zip(data_ncov_d['Long'].astype('float'), data_ncov_d['Lat'].astype('float'))]

gdf_c = GeoDataFrame(data_ncov_c, geometry=geometry_c)   

gdf_r = GeoDataFrame(data_ncov_r, geometry=geometry_r)

gdf_d = GeoDataFrame(data_ncov_d, geometry=geometry_d)



#this is a simple map that goes with geopandas

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))



gdf_c.plot(ax=world.plot(figsize=(15, 10)), marker='o', color='red', markersize=15);

gdf_r.plot(ax=world.plot(figsize=(15, 10)), marker='o', color='green', markersize=15);

gdf_d.plot(ax=world.plot(figsize=(15, 10)), marker='o', color='black', markersize=15);
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



fig, ax = plt.subplots(figsize=(15, 10))



sns.set_color_codes("pastel")

sns.barplot(x=data_ncov_c['2/8/2020 10:04 PM'], y='Province/State', 

            data=data_ncov_c[data_ncov_c['Country/Region']=='Mainland China'][1:], label='Number of cases confirmed on Feb 8, 2020', color='b')



sns.barplot(x=data_ncov_r['2/8/2020 10:04 PM'], y='Province/State', 

            data=data_ncov_r[data_ncov_r['Country/Region']=='Mainland China'][1:], label='Number of cases recovered on Feb 8, 2020', color='g')



sns.barplot(x=data_ncov_d['2/8/2020 10:04 PM'], y='Province/State', 

            data=data_ncov_d[data_ncov_d['Country/Region']=='Mainland China'][1:], label='Number of cases with deaths on Feb 8, 2020', color='black')





# Add a legend and informative axis label

ax.legend(ncol=1, loc='upper right', frameon=True)

ax.set(xlim=(0, 1500), ylabel="", xlabel='Number of cases') # xmax up to ~27000

sns.despine(left=True, bottom=True)
data_ncov_c.head(3)
sum_21 = data_ncov_c.groupby(['Country/Region'])['1/21/2020 10:00 PM'].sum()

sum_22 = data_ncov_c.groupby(['Country/Region'])['1/22/2020 12:00 PM'].sum()

sum_23 = data_ncov_c.groupby(['Country/Region'])['1/23/2020 12:00 PM'].sum()

sum_24 = data_ncov_c.groupby(['Country/Region'])['1/24/2020 12:00 PM'].sum()

sum_25 = data_ncov_c.groupby(['Country/Region'])['1/25/2020 12:00 PM'].sum()

sum_26 = data_ncov_c.groupby(['Country/Region'])['1/26/2020 11:00 PM'].sum() # 11:00 PM !!!

sum_27 = data_ncov_c.groupby(['Country/Region'])['1/27/2020 8:30 PM'].sum()  #  8:30 PM !!!

sum_28 = data_ncov_c.groupby(['Country/Region'])['1/28/2020 11:00 PM'].sum() # 11:00 PM !!!

sum_29 = data_ncov_c.groupby(['Country/Region'])['1/29/2020 9:00 PM'].sum()  #  9:00 PM !!!

sum_30 = data_ncov_c.groupby(['Country/Region'])['1/30/2020 11:00 AM'].sum() # 11:00 AM !!!

sum_31 = data_ncov_c.groupby(['Country/Region'])['1/31/2020 7:00 PM'].sum()  #  7:00 PM !!!



sum_01 = data_ncov_c.groupby(['Country/Region'])['2/1/2020 10:00 AM'].sum()  # 10:00 AM !!!

sum_02 = data_ncov_c.groupby(['Country/Region'])['2/2/2020 9:00 PM'].sum()   #  9:00 PM !!!

sum_03 = data_ncov_c.groupby(['Country/Region'])['2/3/2020 9:00 PM'].sum()   #  9:00 PM !!!

sum_04 = data_ncov_c.groupby(['Country/Region'])['2/4/2020 9:40 AM'].sum()   #  9:40 AM !!!

sum_05 = data_ncov_c.groupby(['Country/Region'])['2/5/2020 11:00 PM'].sum()  # 11:00 PM !!!

sum_06 = data_ncov_c.groupby(['Country/Region'])['2/6/2020 2:20 PM'].sum()   #  2:20 PM !!!

sum_07 = data_ncov_c.groupby(['Country/Region'])['2/7/2020 10:50 PM'].sum()  # 10:50 PM !!!

sum_08 = data_ncov_c.groupby(['Country/Region'])['2/8/20 23:04'].sum()       # 23:04    !!!



sum_21
plt.figure(figsize=(20,6))



ccc = ['Australia','Belgium','Cambodia','Canada','Finland','France','Germany','Hong Kong','India','Italy','Japan','Macau',

       'Mainland China','Malaysia','Nepal','Others','Philippines','Russia','Singapore','South Korea','Spain','Sri Lanka',

       'Sweden','Taiwan','Thailand','UK','US','United Arab Emirates','Vietnam']



ax = plt.gca()

ax.set_yscale('log')

ax.set_xticklabels(ccc)



## Jan

sum_21.plot(marker='')

sum_22.plot(marker='')

sum_23.plot(marker='')

sum_24.plot(marker='')

sum_25.plot(marker='')

sum_26.plot(marker='')

sum_27.plot(marker='')

sum_28.plot(marker='')

sum_29.plot(marker='')

sum_30.plot(marker='')

sum_31.plot(marker='')



## Feb

sum_01.plot(marker='')

sum_02.plot(marker='')

sum_03.plot(marker='')

sum_04.plot(marker='')

sum_05.plot(marker='')

sum_06.plot(marker='')

sum_07.plot(marker='')

sum_08.plot(marker='')



#for l in ax.get_lines():

#    l.remove()