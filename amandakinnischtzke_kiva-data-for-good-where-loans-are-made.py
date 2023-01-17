import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# Load all 4 datasets and check sizes
print("Print dimensions of each dataset...")

kiva_loans = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
print("kiva_loans:")
print(len(kiva_loans),len(kiva_loans.columns))

kiva_mpi = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
print("kiva_mpi: ")
print(len(kiva_mpi),len(kiva_mpi.columns))

loan_theme_ids = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
print("loan_theme_ids:")
print(len(loan_theme_ids),len(loan_theme_ids.columns))

loan_theme_region = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
print("loan_theme_region:")
print(len(loan_theme_region),len(loan_theme_region.columns))

# Get loan counts for each country
data = kiva_loans.groupby(['country_code'])['country_code'].count()
# Reading in separate dataset with [lat,lon] data for all cities & countries
countries = pd.read_csv("../input/countries-latlon/allCountries.csv")
countries = countries.drop(labels=countries.columns[0],axis=1)

# Average geocode data for all cities in a given country to obtain a country 'average' geocode location
country_geocode = countries.groupby(by=countries['country code']).mean()
# Combine loan counts by country with country geocode
new_data = pd.concat([data,country_geocode],ignore_index=False,axis=1)
new_data.columns = ['count','lat','lon']
new_data = new_data.dropna(axis=0)
new_data['count'].describe()
# Make a bubble graph overlaid on world map, to show where the most kiva loans are happening

# Set the dimension of the figure
my_dpi=96
plt.figure(figsize=(2600/my_dpi, 1800/my_dpi), dpi=my_dpi)

# Make the background map
m=Basemap(llcrnrlon=-180, llcrnrlat=-65,urcrnrlon=180,urcrnrlat=80)
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='grey', alpha=0.3)
m.drawcoastlines(linewidth=0.1, color="white")
 
# prepare a color for each point depending on the continent.
new_data['labels_enc'] = pd.factorize(new_data.index)[0]

# Add a point per position
m.scatter(new_data['lon'], new_data['lat'],s=new_data['count']/6, alpha=0.4, c=new_data['labels_enc'], cmap="Set1")

# copyright and source data info
plt.text( -170, -58,'Where kiva loans are made', ha='left', va='bottom', size=20, color='#555555' )
 
# Save as png

