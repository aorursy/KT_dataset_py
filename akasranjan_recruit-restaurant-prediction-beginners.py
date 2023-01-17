# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

avd = pd.read_csv('../input/air_visit_data.csv')

asi = pd.read_csv('../input/air_store_info.csv')

hsi = pd.read_csv('../input/hpg_store_info.csv')

ar = pd.read_csv('../input/air_reserve.csv')

hr = pd.read_csv('../input/hpg_reserve.csv')

sid = pd.read_csv('../input/store_id_relation.csv')

tes = pd.read_csv('../input/sample_submission.csv')

hol = pd.read_csv('../input/date_info.csv')

plt.rcParams['figure.figsize'] = 16, 8
len(avd)
hol.head()
avd.head()
air_visit_date = pd.merge(avd, hol, how='left', left_on='visit_date', right_on='calendar_date')
air_visit_date.head()
air_visit_date.loc[air_visit_date['holiday_flg'] != 0].sort_values('visitors', ascending=False).head(10)
day_wise_df = air_visit_date.loc[air_visit_date['holiday_flg'] != 0].groupby('day_of_week').agg(sum)
sorter = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

sorterIndex = dict(zip(sorter,range(len(sorter))))

day_wise_df['day_id'] = day_wise_df.index

day_wise_df['day_id'] = day_wise_df['day_id'].map(sorterIndex)
day_wise_df.plot()

day_wise_df.plot(kind='bar')
day_wise_df
day_wise_df.head()
day_wise_df.sort_values('day_id', inplace=True)
day_wise_df
air_visit_date.loc[air_visit_date['holiday_flg'] == 0].sort_values('visitors', ascending=False).head(10)
day_wise_no_holiday_df = air_visit_date.loc[air_visit_date['holiday_flg'] == 0].groupby('day_of_week').agg(sum)
day_wise_no_holiday_df
day_wise_no_holiday_df['day_id'] = day_wise_no_holiday_df.index

day_wise_no_holiday_df['day_id'] = day_wise_no_holiday_df['day_id'].map(sorterIndex)
day_wise_no_holiday_df.sort_values('day_id', inplace=True)
day_wise_no_holiday_df
day_wise_no_holiday_df.plot()

day_wise_no_holiday_df.plot(kind='bar')