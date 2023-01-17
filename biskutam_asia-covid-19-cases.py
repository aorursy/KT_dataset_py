import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # chart
asia_df = pd.read_csv('../input/covid19-cases-in-africa/covid19_asia.csv')
# import covid19_asia.csv as initial input
asia_df.head()
asia_df.describe()
df_group_country_date = asia_df.groupby( [ "Country_Region", "ObservationDate"] ).count().reset_index()
df_group_country = asia_df.groupby( [ "Country_Region"] ).count().reset_index()

df_group_country_date = df_group_country_date.drop([ 'Province_State'], axis = 1)
df_group_country = df_group_country.drop(['ObservationDate', 'Province_State'], axis = 1)
# Overview of the new dataframe
df_group_country_date.head()
# Overview of the new dataframe
df_group_country.head()
df_group_country.sort_values(by=['Active'], ascending=False)
asia_df.sort_values(by=['Active','ObservationDate'], ascending=False)
asia_df.isnull().sum()
# check if column have null value
# Province_State got many null value, and Action
null_df = asia_df[asia_df.isna().any(axis=1)]
null_df.shape

# show the shape of null_df, basically the null value is correct
null_df.head()
