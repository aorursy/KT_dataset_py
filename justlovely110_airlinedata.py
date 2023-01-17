# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib as plt
# 1. Combine different csv files into a single dataframe
files = []
files.append(pd.read_csv("../input/airline-2019/Jan2019/405557996_T_ONTIME_REPORTING.csv"))
files.append(pd.read_csv("../input/airline-2019/apr2019/405557996_T_ONTIME_REPORTING.csv"))
files.append(pd.read_csv("../input/airline-2019/aug2019/405557996_T_ONTIME_REPORTING.csv"))
files.append(pd.read_csv("../input/airline-2019/dec20219/405557996_T_ONTIME_REPORTING.csv"))
files.append(pd.read_csv("../input/airline-2019/feb2019/405557996_T_ONTIME_REPORTING.csv"))
files.append(pd.read_csv("../input/airline-2019/jul2019/405557996_T_ONTIME_REPORTING.csv"))
files.append(pd.read_csv("../input/airline-2019/june2019/405557996_T_ONTIME_REPORTING.csv"))
files.append(pd.read_csv("../input/airline-2019/mar2019/405557996_T_ONTIME_REPORTING.csv"))
files.append(pd.read_csv("../input/airline-2019/may2019/405557996_T_ONTIME_REPORTING.csv"))
files.append(pd.read_csv("../input/airline-2019/nov2019/405557996_T_ONTIME_REPORTING.csv"))
files.append(pd.read_csv("../input/airline-2019/oct2019/405557996_T_ONTIME_REPORTING.csv"))
files.append(pd.read_csv("../input/airline-2019/sept2019/405557996_T_ONTIME_REPORTING.csv"))
df = pd.concat(files)
# 2. Clean the city_name columns, which also contain the abreviated state names. 
df.ORIGIN_CITY_NAME = df.ORIGIN_CITY_NAME.apply(lambda x: x[:-4])
df.DEST_CITY_NAME = df.DEST_CITY_NAME.apply(lambda x: x[:-4])
# 3. Check which of the columns are redundant information (i.e. they can easily be computed from the other columns)
df.dropna(0)
df["TOTAL_DELAY"] = df.CARRIER_DELAY + df.WEATHER_DELAY + df.NAS_DELAY + df.SECURITY_DELAY + df.LATE_AIRCRAFT_DELAY
# 4.Find out the airports and the flight operators which correspond to maximum delay in general. 
## These destination airports correspond to the most delay in general.
print(df.pivot_table(index=["DEST","DEST_CITY_NAME",'DEST_STATE_NM'],values="TOTAL_DELAY",aggfunc='mean').sort_values("TOTAL_DELAY",ascending=False).head(10))
# 4.Find out the airports and the flight operators which correspond to maximum delay in general. 
## These origin airports correspond to the most delay in general.
print(df.pivot_table(index=["ORIGIN","ORIGIN_CITY_NAME",'ORIGIN_STATE_NM'],values="TOTAL_DELAY",aggfunc='mean').sort_values("TOTAL_DELAY",ascending=False).head(10))
# 4.Find out the airports and the flight operators which correspond to maximum delay in general. 
## These airline operators correspond to the most delay in general.
print(df.pivot_table(index=["OP_CARRIER_AIRLINE_ID"],values="TOTAL_DELAY",aggfunc='mean').sort_values("TOTAL_DELAY",ascending=False).head(10))