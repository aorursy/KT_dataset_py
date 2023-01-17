# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# custom import

import geopandas as gpd

import matplotlib.pyplot as plt

%matplotlib inline
# Read the map data

country_provinces = gpd.read_file("../input/ph-provinces/Provinces.shp")



# country_provinces.loc[(country_provinces['PROVINCE']=='North Cotabato')]

country_provinces.iloc[52,:]
type(country_provinces)
# select a subset

data = country_provinces.loc[:,['PROVINCE','REGION','geometry']]
data.plot();
production_data = pd.read_csv('../input/production/production.csv',na_values=['..'],thousands=',')

production_data.shape
production_data['Area'] = production_data.Area.replace({'\..':''},regex=True)

production_data.sample(5)
production_data.dtypes
production_data.columns = ['commodity','area_type','PROVINCE','year','production']

production_data.sample(5)
palay_latest = production_data.loc[(production_data['commodity']=='Corn') & (production_data['year']==2019) & (production_data['area_type']=='Provincial')]

palay_latest.sample(5)
country_provinces = country_provinces.merge(palay_latest,on='PROVINCE')
country_provinces
country_provinces.plot('production', figsize=(15, 15), legend=True)

plt.title('Corn Production')