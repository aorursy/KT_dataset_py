import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
# uninteresting columns
DROP_COLS = ["driver_age_raw",'search_type_raw','id']
df = pd.read_csv('../input/SC.csv',parse_dates=['stop_date'],infer_datetime_format=True)
print("# rows:",df.shape[0])  # RAW data has : 8,440,935 rows
print("\n Raw: # columns:",df.shape[1])
df.dropna(how="all",axis=1,inplace=True)
print("\n # columns with values:",list(df.columns))
print("\n nunique:", df.nunique())
df.head()
print(df.shape)
print("\n nunique:", df.nunique())
df.drop(DROP_COLS,axis=1,inplace=True)
## Drop all nan columns. Could drop unary column (State)
# df.dropna(how="all",axis=1,inplace=True)
df.shape
df.isna().sum()
df.dropna(subset=["stop_date","driver_gender","stop_purpose","driver_race_raw","driver_race","driver_age","location_raw",
                  "officer_id", "officer_race","county_fips","stop_outcome","is_arrested","road_number","police_department"],inplace=True)
df.shape

df.sample(n=112345).to_csv("SC_trafficStops_v1_100k.csv.gz",index=False,compression="gzip")