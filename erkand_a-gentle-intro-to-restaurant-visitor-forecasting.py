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
air_reserve = pd.read_csv("../input/air_reserve.csv")

print("# of air_reserve entries:", len(air_reserve))

air_reserve.head()
# Check if data includes null entries

air_reserve.info(verbose=True, null_counts=True)
# some statistics about the visitor based on air_reserve data

air_reserve.describe()
# boxplot is useful to see the distribution of reservation numbers

air_reserve.boxplot(figsize=(15,10))
hpg_reserve = pd.read_csv("../input/hpg_reserve.csv")

print("# of hpg_reserve entries:", len(hpg_reserve))

hpg_reserve.head()
# Check if data includes null entries

# if you do not include null_counts=True, info does only shows column data types

# (I believe this behaviour depends on the number of dataframe entries)

hpg_reserve.info(verbose=True, null_counts=True) 
hpg_reserve.describe()
hpg_reserve.boxplot(figsize=(15, 10))
date_info = pd.read_csv("../input/date_info.csv")

date_info.head(11)
# If we look at the date_info data, we will see that holidays are flagged with 1, the rest is 

# flagged with 0

date_info.holiday_flg.unique()
date_info.info()
# convert type of the calendar_date column to datetime type

date_info['calendar_date'] = pd.to_datetime(date_info['calendar_date'])

# convert type of day_of_week column to type to category

date_info['day_of_week'] = date_info['day_of_week'].astype('category')
date_info.info()
# convert visit_datetime and reserve_datetime to datatime data type

air_reserve['visit_datetime'] = pd.to_datetime(air_reserve['visit_datetime'])

air_reserve['reserve_datetime'] = pd.to_datetime(air_reserve['reserve_datetime'])

# merge the date_info with air_reserve by joining on the date, in this way, 

# each date in air_reserve dataframe will have a day of a week and holiday flg information

air_reserve_days = pd.merge(air_reserve, date_info[['day_of_week', 'holiday_flg']],

                            left_on=air_reserve['visit_datetime'].dt.date,

                            right_on=date_info['calendar_date'].dt.date,

                            left_index=True)
air_reserve_days.head(10)
gbo = air_reserve_days.groupby(['day_of_week'])
gbo.groups.keys()
# it is not surprising that we have restaurant visits most on Friday and Saturday

# and least on Monday

# beware of that the following data is only based on air_reserve and includes all

# restaurant reservations

gbo['reserve_visitors'].sum()
gbo['reserve_visitors'].sum().plot(kind='bar', 

                                   title =' week days vs. # of restaurant visit (air_reserve)',

                                   grid=True,

                                   figsize=(15,8))
hpg_reserve.head()
# convert visit_datetime and reserve_datetime to datatime data type

hpg_reserve['visit_datetime'] = pd.to_datetime(hpg_reserve['visit_datetime'])

hpg_reserve['reserve_datetime'] = pd.to_datetime(hpg_reserve['reserve_datetime'])

# merge the date_info with hpg_reserve by joining on the date, in this way, 

# each date in hpg_reserve dataframe will have a day of a week and holiday flg information

hpg_reserve_days = pd.merge(hpg_reserve, date_info[['day_of_week', 'holiday_flg']],

                            left_on=hpg_reserve['visit_datetime'].dt.date,

                            right_on=date_info['calendar_date'].dt.date,

                            left_index=True)
hpg_reserve_days.head()
gbo_hpg = hpg_reserve_days.groupby(['day_of_week'])
gbo_hpg.groups.keys()
gbo_hpg['reserve_visitors'].sum()
(gbo_hpg['reserve_visitors'].sum()).plot(kind='bar', 

                                   title =' week days vs. # of restaurant visit (hpg_reserve)',

                                   grid=True,

                                   figsize=(15,8))
air_visit_data = pd.read_csv("../input/air_visit_data.csv")

print("# of air_visit_data entries: ", len(air_visit_data))

air_visit_data.head(10)
air_visit_data.plot(kind='line', x='visit_date', y='visitors', grid=True,

                    alpha=0.5, color='g', figsize=(15, 8))
# convert visit_datetime and reserve_datetime to datatime data type

air_visit_data['visit_date'] = pd.to_datetime(air_visit_data['visit_date'])

# merge the date_info with hpg_reserve by joining on the date, in this way, 

# each date in hpg_reserve dataframe will have a day of a week and holiday flg information

air_visit_data_days = pd.merge(air_visit_data, 

                               date_info[['day_of_week', 'holiday_flg']],

                               left_on=air_visit_data['visit_date'].dt.date,

                               right_on=date_info['calendar_date'].dt.date,

                               left_index=True)
air_visit_data_days.head()
gbo_air_visit = air_visit_data_days.groupby(['day_of_week'])

gbo_air_visit.groups.keys()
gbo_air_visit['visitors'].sum()
(gbo_air_visit['visitors'].sum()).plot(kind='bar', 

                                   title =' week days vs. # of restaurant visit (air_visit)',

                                   grid=True,

                                   figsize=(15,8))
store_id_relation = pd.read_csv("../input/store_id_relation.csv")

print("# of store_id_relation: ", len(store_id_relation))

store_id_relation.head()
air_store_info = pd.read_csv("../input/air_store_info.csv")

print("# of air_store_info entries:", len(air_store_info))

air_store_info.head(10)
hpg_store_info = pd.read_csv("../input/hpg_store_info.csv")

print("# of hpg_store_info entries:", len(hpg_store_info))

hpg_store_info.head()
# The test set covers the last week of April and May of 2017. 

# 8 days from April, 31 days from May, in total: 39 days * # of restaurants

sample_submission = pd.read_csv("../input/sample_submission.csv")

print("# of sample_submission entries: ", len(sample_submission))

sample_submission.head()

print("# of unique restaurants considered : ", 32019/39)