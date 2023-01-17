# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")
sf_permits_desc = pd.read_excel("../input/building-permit-applications-data/DataDictionaryBuildingPermit.xlsx")

# set seed for reproducibility
np.random.seed(0) 
# look at the first 5 rows of the nfl_data file.
print("nfl_data has {} rows and {} columns".format(*nfl_data.shape))
nfl_data.head()
print("sf_permits has {} rows and {} columns".format(*sf_permits.shape))
sf_permits.head()
def how_many_missing_data_points(df):
    # get the number of missing data points per column
    df_missing_values = pd.DataFrame(df.isnull().sum(), columns=["Missing Counts"])
    df_missing_values["Missing Ratio"] = df_missing_values["Missing Counts"] / df.shape[0]
    # look at the # of missing points in the top ten columns
    print("The 10 columns which have the most missing points")
    display(df_missing_values.sort_values(by="Missing Counts", ascending=False)[0:10])
    # get the number of missing data points per column
    total_cells = np.product(df.shape)
    total_missing = df_missing_values["Missing Counts"].sum()
    # percent of data that is missing
    print("Missing Ratio: {:.2%}".format(total_missing/total_cells))
    return df_missing_values
df_nfl_data_missing_values = how_many_missing_data_points(nfl_data)
df_sf_permits_missing_values = how_many_missing_data_points(sf_permits)
df_nfl_data_missing_values.loc[["TimeSecs"]]
cols = ["Street Number Suffix", "Zipcode"]
# Show the description of the two columns
display(sf_permits_desc[sf_permits_desc["Column name"].isin(cols)])
display(df_sf_permits_missing_values.loc[cols, :])
display(sf_permits.loc[sf_permits["Street Number Suffix"].notnull(), ["Street Number Suffix"]].head())
display(sf_permits.loc[sf_permits["Zipcode"].notnull(), ["Zipcode"]].head())
def drop_missing_columns(df):
    print("Columns in original dataset: %d \n" % df.shape[1])
    df_dropped = df.dropna(axis=1)
    print("Columns with na's dropped: %d" % df_dropped.shape[1])
    return df_dropped
# remove all columns with at least one missing value
nfl_data_with_na_dropped = drop_missing_columns(nfl_data)
nfl_data_with_na_dropped.head()
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna("0")
from sklearn.preprocessing import Imputer
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head(8)
subset_nfl_data
# Fill it with the mean number
imputer_mean = Imputer()
pd.DataFrame(imputer_mean.fit_transform(subset_nfl_data), columns=subset_nfl_data.columns, index=subset_nfl_data.index)
# Fill it with the median number
imputer_median = Imputer(strategy="median",axis=0)
pd.DataFrame(imputer_median.fit_transform(subset_nfl_data), columns=subset_nfl_data.columns, index=subset_nfl_data.index)
# Fill it with the mode number
imputer_mode = Imputer(strategy="most_frequent",axis=0)
pd.DataFrame(imputer_mode.fit_transform(subset_nfl_data), columns=subset_nfl_data.columns, index=subset_nfl_data.index)
