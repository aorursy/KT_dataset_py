import pandas as pd

import matplotlib.pyplot as plt

import geopandas as gpd
shapefile = '/kaggle/input/us-district-court-jurisdictions/US_District_Court_Jurisdictions.shp'

map_df = gpd.read_file(shapefile)

map_df.head(5)
cus_map_df = map_df[~map_df.FID.isin([4,22,69,82,93,94])]
cus_map_df.plot(figsize=[20,20])
# Read in the population csv file.

district_population = pd.read_csv('/kaggle/input/us-district-court-jurisdictions/district_population.csv')

district_population['key'] = district_population['district_name'].apply(lambda x: x.upper())
# Join with the geopandas dataframe, dropping rows without population values.

pop_df = map_df.join(district_population.set_index('key'), on='DISTRICT').dropna()

pop_df.head(5)
# Create a plot with the population data.



variable = 'census_2010_population'

vmin, vmax = 0.67, 9.2

fig, ax = plt.subplots(1, figsize=(10, 6))



pop_df.plot(column=variable, cmap='BuGn', linewidth=0.8, ax=ax, edgecolor='0.8')



ax.axis('off')

ax.set_title('US District Court Population (Millions) ', fontdict={'fontsize': '21', 'fontweight' : '3'})

ax.annotate('Source: US Census & PACER, 2010',xy=(0.1, .08),  

            xycoords='figure fraction', 

            horizontalalignment='left', 

            verticalalignment='top', 

            fontsize=12, 

            color='#555555')

            



sm = plt.cm.ScalarMappable(cmap='BuGn', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)