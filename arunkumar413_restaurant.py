# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
air_reserve = pd.read_csv("../input/air_reserve.csv", parse_dates=["visit_datetime", "reserve_datetime"])

air_store_info = pd.read_csv("../input/air_store_info.csv")

air_visit_data = pd.read_csv("../input/air_visit_data.csv")

date_info = pd.read_csv("../input/date_info.csv")



hpg_reserve = pd.read_csv("../input/hpg_reserve.csv")

hpg_store_info = pd.read_csv("../input/hpg_store_info.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")

store_id_relation = pd.read_csv("../input/store_id_relation.csv")

#air_reserve.head()
#check the dtypes



air_reserve.dtypes
air_store_info.head()
#merge the reserve data frames

air_merged= air_reserve.merge(air_store_info, on="air_store_id",how='left')

air_merged.head()
# split the datetime field to create more granualr fields for better analysis



air_merged["visit_hour"] = air_merged["visit_datetime"].dt.hour

air_merged["vist_day"] = air_merged["visit_datetime"].dt.day

air_merged["vist_month"] = air_merged["visit_datetime"].dt.month

air_merged["vist_year"] = air_merged["visit_datetime"].dt.year

air_merged["visit_day_name"] = air_merged["visit_datetime"].dt.weekday_name

air_merged["visit_date"] = air_merged["visit_datetime"].dt.date



air_merged["reserve_hour"] = air_merged["reserve_datetime"].dt.hour

air_merged["reserve_day"] = air_merged["reserve_datetime"].dt.day

air_merged["reserve_month"] = air_merged["reserve_datetime"].dt.month

air_merged["reserve_year"] = air_merged["reserve_datetime"].dt.year

air_merged["reserve_day_name"] = air_merged["reserve_datetime"].dt.weekday_name

air_merged["reserve_date"] = air_merged["reserve_datetime"].dt.date

#get the list of holidays



holidays = date_info[date_info["holiday_flg"]==1]

holidays = pd.DataFrame(holidays)

holidays.dtypes
holidays["calendar_date"]= holidays["calendar_date"].astype("datetime64[ns]")

air_merged["visit_date"] = air_merged["visit_date"].astype("datetime64[ns]")
air_merged= air_merged.assign(holiday= air_merged["visit_date"].isin(holidays["calendar_date"]))

air_merged.tail()