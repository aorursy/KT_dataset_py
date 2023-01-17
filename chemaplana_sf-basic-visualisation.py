# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
sf_file = pd.read_csv("../input/Restaurant_Scores_-_LIVES_Standard.csv")
sf_rev = sf_file.drop(['business_address', 'business_city', 'business_state',

	'business_postal_code', 'business_location', 'business_phone_number'], axis=1)

sf_rev['inspection_date'] = pd.to_datetime(sf_rev['inspection_date'], 

                                           format='%m/%d/%Y %H:%M:%S %p')

sf_rev['year'] = sf_rev['inspection_date'].apply(lambda x: x.year)

sf_rev['month'] = sf_rev['inspection_date'].apply(lambda x: x.month)
sf_rev_dum = pd.get_dummies(sf_rev['risk_category'], prefix='Risk')

sf_rev = pd.concat([sf_rev, sf_rev_dum], axis = 1, join='inner')
sf_rev_grouped = sf_rev.loc[:,['inspection_id','business_latitude','business_longitude','year', 'month',

	'Risk_High Risk','Risk_Low Risk','Risk_Moderate Risk']]

sf_rev_grouped = sf_rev_grouped.groupby(['inspection_id','business_latitude','business_longitude', 'year', 'month'], as_index=False).sum()



col_names = ['inspection_id','business_latitude','business_longitude', 'year', 'month',

	'Tot_High_Risk','Tot_Low_Risk','Tot_Moderate_Risk']

sf_rev_grouped.columns = col_names
import matplotlib.pyplot as plt

import geopandas as gpd



fig, ax = plt.subplots()

ax.set_aspect('equal')

ax.set_axis_bgcolor('paleturquoise')

ax.set_title('3 or more High Risk observations in SF restaurants per inspection per year \n by supervisorial district')

xline = np.linspace(-122.3,-122.5,5)

yline = np.linspace(37.7,37.85,4)

ax.set_xticks(xline)

ax.set_xticklabels(xline)

ax.set_yticks(yline)

ax.set_yticklabels(yline)



shp_file1 = gpd.read_file('../input/Shapefiles (2)/geo_export_38fd3153-0303-488e-9f3c-0f81e8e00115.shp')



shp_file1['coords'] = shp_file1['geometry'].apply(lambda x: x.representative_point().coords[:])

shp_file1['coords'] = [coords[0] for coords in shp_file1['coords']]



shp_file1.plot(ax=ax, column='supervisor', cmap='Spectral')

for idx, row in shp_file1.iterrows():

	plt.annotate(s=row['numbertext'], xy=row['coords'],

		horizontalalignment='center',

		verticalalignment='top')

    

sf_rev_grouped = sf_rev_grouped[sf_rev_grouped.business_longitude <= -122.3]

sf_rev_grouped = sf_rev_grouped[sf_rev_grouped.business_latitude >= 37.7]

sf_rev_high = sf_rev_grouped[sf_rev_grouped.Tot_High_Risk >= 3]



sf_rev_high_2014 = sf_rev_high[sf_rev_high.year == 2014]

plt.scatter(sf_rev_high_2014['business_longitude'], sf_rev_high_2014['business_latitude'], 

	marker='o', color='blue', s=sf_rev_high_2014['Tot_High_Risk']*10, alpha=1.0, 

            label='2014 inspections')

sf_rev_high_2015 = sf_rev_high[sf_rev_high.year == 2015]

plt.scatter(sf_rev_high_2015['business_longitude'], sf_rev_high_2015['business_latitude'], 

	marker='o', color='red', s=sf_rev_high_2015['Tot_High_Risk']*10, alpha=1.0, 

            label='2015 inspections')

plt.legend(loc='upper left')



plt.show()