import pandas as pd 

import geopandas as gpd

import matplotlib.pyplot as plt 
df = pd.read_excel('../input/paramedical-staff-in-india/paramed/paramedical_staff.xlsx')

df.head()
shp_gdf = gpd.read_file('../input/india-gis-data/India States/Indian_states.shp')

shp_gdf.head()
merged = shp_gdf.set_index('st_nm').join(df.set_index('States'))

merged.head()
fig, ax = plt.subplots(1, figsize=(12, 12))

ax.axis('off')

ax.set_title('Paramedical Staffs at District Hospitals in India as of 31st March 2019',

             fontdict={'fontsize': '15', 'fontweight' : '3'})

fig = merged.plot(column='Staff', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)