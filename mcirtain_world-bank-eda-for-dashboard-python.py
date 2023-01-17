import datetime # for filtering on open calls
import geopandas as gpd # for mapping - learn at https://towardsdatascience.com/lets-make-a-map-using-geopandas-pandas-and-matplotlib-to-make-a-chloropleth-map-dddc31c1983d
import matplotlib.pyplot as plt
import pandas as pd # for reading/munging our data

%matplotlib inline

# read data
df = pd.read_csv('../input/procurement-notices/procurement-notices.csv')
df.head()

# clean up column names:
cols = df.columns
new_cols = []

for name in cols:
    #remove leading/trailing whitespace, replace remaining spaces with underscore
    new_name = name.lstrip().rstrip().lower().replace(' ','_') 
    new_cols.append(new_name)
    
df.columns = new_cols
df.columns
# Handle field types, and filter on deadline dates not yet passed:   
df.dtypes # I see we'll need to convert date fields to date type.
df.publication_date = pd.to_datetime(df['publication_date'])
df.deadline_date = pd.to_datetime(df['deadline_date'])

# replace missing deadline dates with tomorrow's date?  
df.loc[df.deadline_date.isna(), 'deadline_date'] = datetime.date.today() + datetime.timedelta(days=1) 
df[df.deadline_date >= pd.to_datetime('today')].shape[0] # count rows where due date greater than today
# Setup Mapping (see linked tutorial from import of gpd)
# load in a shapefile (from https://www.arcgis.com/home/item.html?id=2ca75003ef9d477fb22db19832c9554f)
shp = '../input/countries-shapefile/countries.shp'
map_df = gpd.read_file(shp)
# check data type so we can see that this is not a normal dataframe, but a GEOdataframe
map_df.head()
map_df.plot() #WOW!
# join geodata (map_df) and dataset (df)
merged = map_df.merge(df, how='right', left_on='NAME', right_on='country_name')
merged[merged.country_name.isna()].NAME.value_counts()
df_clean = merged[pd.notnull(merged['country_name'])]
df_clean[df_clean.country_name.isna()].NAME.value_counts()
merged['country_name'].fillna(merged['NAME'])
merged[merged.country_name.isna()].NAME.value_counts()

# plot number of bids by country

# set the range for the choropleth
vmin, vmax = 120, 220
# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(16, 16))

# create map
merged.plot(column='country_name', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
#merged.plot(column='SOVEREIGN', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
df[df['deadline_date']>pd.to_datetime('today')].groupby('deadline_date').id.count().plot()
# another approach:  https://www.kaggle.com/ashokphili/world-bank-procurement/notebook