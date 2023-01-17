import  pandas as pd 
import numpy as np
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv");
sf_permits=pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv");


np.random.seed(0)
nfl_data.sample(5)
sf_permits.sample(5)
missing_value_count_nfl_data=nfl_data.isnull().sum()
missing_value_count_nfl_data
missing_value_count_nfl_data[40:50]
nfl_data.shape
total_cells_with_missing_Data_nfl_data=np.product(nfl_data.shape)
missing_value_count_nfl_data.sum()
total_missing_value=missing_value_count_nfl_data.sum()
Total_missing_value_percentage=(total_missing_value/total_cells_with_missing_Data_nfl_data)*100
Total_missing_value_percentage
missing_value_count_sf_permit_data=sf_permits.isnull().sum()
missing_value_count_sf_permit_data
total_cell_sf_permit=np.product(sf_permits.shape)
total_cell_sf_permit
total_missing_value_sf_permit=missing_value_count_sf_permit_data.sum()
total_missing_value_sf_permit
total_missing_value_percentage_sf_permit=(total_missing_value_sf_permit/total_cell_sf_permit)*100
total_missing_value_percentage_sf_permit
nfl_data
columns_with_na_dropped_nfl_data = nfl_data.dropna(axis=1)
columns_with_na_dropped_nfl_data.head()
sf_permits.dropna()
sf_permits
columns_with_na_dropped_sf_permit=sf_permits.dropna(axis=1)
columns_with_na_dropped_sf_permit
subset_nfl_data=nfl_data.loc[:, 'EPA':'Season']
subset_nfl_data
subset_nfl_data.fillna(0)
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
sf_permits

columns_with_na_dropped_sf_permit
columns_with_na_dropped_sf_permit.fillna(method = 'bfill', axis=0).fillna("0")
sf_permits.fillna(method = 'bfill', axis=0).fillna("0")
