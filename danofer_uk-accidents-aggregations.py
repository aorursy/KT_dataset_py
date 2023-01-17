import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
cols_keep = ['Accident_Severity', 'Date','Time', 'Latitude','Longitude',
             'Local_Authority_(District)', 'Local_Authority_(Highway)',
            'LSOA_of_Accident_Location', 'Number_of_Casualties', "1st_Road_Number","2nd_Road_Number"]
df = pd.read_csv('../input/Accident_Information.csv',usecols=cols_keep, #nrows=12345,
                 parse_dates=[['Date', 'Time']],keep_date_col=True)
df.shape
df["Date_Time"] = pd.to_datetime(df["Date_Time"],infer_datetime_format=True,errors="coerce")
# we see that some cases lack a time of events - creating a bad date format. we'll fix these

df.loc[df['Date_Time'].isna(), 'Date_Time'] = df["Date"]
df.loc[df["Date_Time"].isna()]
df.drop(["Date","Time"],axis=1,inplace=True)
df.set_index("Date_Time",inplace=True)
df.index = pd.to_datetime(df.index)
df["serious_accident"] = df.Accident_Severity != "Slight"
df.nunique()
df.columns
df.describe()
df.index.dtype
df.head()
# Identifying the worst districts to travel.
### https://stackoverflow.com/questions/19384532/how-to-count-number-of-rows-per-group-and-other-statistics-in-pandas-group-by
### https://stackoverflow.com/questions/32012012/pandas-resample-timeseries-with-groupby/39186403#39186403

lsoa_wise = df.groupby( 'LSOA_of_Accident_Location').resample("M").agg({"Number_of_Casualties":"sum","serious_accident":"sum",
                                                                        "Accident_Severity":"count",
                                                                       
#                                                                         "Latitude":scipy.stats.mode,"Longitude":scipy.stats.mode
#                                                                         "Latitude":"mean","Longitude":"mean" # we get missing latLong when no accidents occured, and their locations can change unless we use mode! 
                                                                       })
lsoa_wise.rename(columns={"Accident_Severity":"Accident_counts"},inplace=True)
lsoa_wise["percent_seriousAccidents"] = 100*lsoa_wise["serious_accident"]/lsoa_wise["Accident_counts"].round(2)
lsoa_wise.loc[lsoa_wise['percent_seriousAccidents'].isna(), 'percent_seriousAccidents'] = 0
print(lsoa_wise.shape)
lsoa_wise.head()
lsoa_wise.describe()
lsoa_wise.to_csv("uk_accidents_lsoa_monthly.csv.gz",compression="gzip")