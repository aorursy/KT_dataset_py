# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

addresses_df = pd.read_csv('../input/paradise-papers/paradise_papers.nodes.address.csv')
lat_long_df = pd.read_csv('../input/paradise-papers-us-geocoded/paradise_papers_US_geocoded.csv',encoding = "ISO-8859-1")#paradise_papers_US_geocoded.csv

us_addresses = addresses_df[addresses_df['n.country_codes']=='USA']
us_addresses.head()

us_addresses_compact = us_addresses[['n.node_id','n.name']]
us_addresses_compact
#https://stackoverflow.com/questions/25292838/applying-regex-to-a-pandas-dataframe
import re
def split_it(address):
    toReturn = re.findall('\d\d\d\d\d', str(address))
    if len(toReturn) ==0:
        toReturn = ['NaN']
    return toReturn[0]

#us_addresses_compact['zip_code']
us_addresses_compact['zip_code'] = us_addresses_compact['n.name'].apply(split_it)

us_addresses_compact
zip_counts = us_addresses_compact['zip_code'].value_counts()
zip_counts.sort_values
zip_counts[1:].head(20).plot.bar()
plt.xticks(rotation=45, fontsize=12)
plt.yticks( fontsize=12)
plt.title('Paradise Papers \n Address Count by ZIP (US)', fontsize = 20)
plt.ylabel('Address Count',fontsize=15)
plt.xlabel('Top 20 US Zip Codes \n 100** is Manhattan',fontsize=15)
us_addresses_compact[us_addresses_compact['zip_code']=='10022']
hot_zip =us_addresses_compact[us_addresses_compact['zip_code']=='10022']
hot_zip[hot_zip['n.name'].str.contains('1000')] #Looking for Bloomingdale's (1000 Third Avenue). Not found
hot_zip[hot_zip['n.name'].str.contains('14 ')] #Looking for St. Patrick's (14 East 51st street). St. Not found.
hot_zip[hot_zip['n.name'].str.contains('885')]#Looking for the Lipstick Building (885 Third Avenue). 7 hits.
us_addresses_compact.head()
lat_long_compact = lat_long_df[['node_id','lat','long']].dropna()
temp = us_addresses_compact.merge(lat_long_compact, left_on='n.node_id', right_on='node_id', how='outer' )
us_addresses_lat_long_compact = temp.dropna().drop(columns=['node_id'])
us_addresses_lat_long_compact
#from mpl_toolkits.basemap import Basemap
#from matplotlib import cm
#%matplotlib inline
 
#west, south, east, north = -74.26, 40.50, -73.70, 40.92
 
#fig = plt.figure(figsize=(14,10))
#ax = fig.add_subplot(111)
 
#m = Basemap( llcrnrlat=south, urcrnrlat=north,
#            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')
#x, y = m(us_addresses_lat_long_compact['long'].values, us_addresses_lat_long_compact['lat'].values)
#m.hexbin(x, y, gridsize=1000,
 #        bins='log', cmap=cm.YlOrRd_r);
