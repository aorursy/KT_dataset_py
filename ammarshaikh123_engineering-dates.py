# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



import datetime
# let's load the Lending Club dataset with a few selected columns

# just a few rows to speed things up



use_cols = ['issue_d', 'last_pymnt_d']

data = pd.read_csv('../input/loan.csv', usecols=use_cols, nrows=10000)

data.head()
# now let's parse the dates, currently coded as strings, into datetime format



data['issue_dt'] = pd.to_datetime(data.issue_d)

data['last_pymnt_dt'] = pd.to_datetime(data.last_pymnt_d)



data[['issue_d','issue_dt','last_pymnt_d', 'last_pymnt_dt']].head()
# Extracting Month from date



data['issue_dt_month'] = data['issue_dt'].dt.month



data[['issue_dt', 'issue_dt_month']].head()
data[['issue_dt', 'issue_dt_month']].tail()
# Extract quarter from date variable



data['issue_dt_quarter'] = data['issue_dt'].dt.quarter



data[['issue_dt', 'issue_dt_quarter']].head()
data[['issue_dt', 'issue_dt_quarter']].tail()
# We could also extract semester



data['issue_dt_semester'] = np.where(data.issue_dt_quarter.isin([1,2]),1,2)

data.head()
# day - numeric from 1-31



data['issue_dt_day'] = data['issue_dt'].dt.day



data[['issue_dt', 'issue_dt_day']].head()
# day of the week - from 0 to 6



data['issue_dt_dayofweek'] = data['issue_dt'].dt.dayofweek



data[['issue_dt', 'issue_dt_dayofweek']].head()
data[['issue_dt', 'issue_dt_dayofweek']].tail()
# day of the week - name



data['issue_dt_dayofweek'] = data['issue_dt'].dt.weekday_name



data[['issue_dt', 'issue_dt_dayofweek']].head()
data[['issue_dt', 'issue_dt_dayofweek']].tail()
# was the application done on the weekend?



data['issue_dt_is_weekend'] = np.where(data['issue_dt_dayofweek'].isin(['Sunday', 'Saturday']), 1,0)

data[['issue_dt', 'issue_dt_dayofweek','issue_dt_is_weekend']].head()
data[data.issue_dt_is_weekend==1][['issue_dt', 'issue_dt_dayofweek','issue_dt_is_weekend']].head()
# extract year 



data['issue_dt_year'] = data['issue_dt'].dt.year



data[['issue_dt', 'issue_dt_year']].head()
# perhaps more interestingly, extract the date difference between 2 dates



data['issue_dt'] - data['last_pymnt_dt']

# same as above capturing just the time difference

(data['last_pymnt_dt']-data['issue_dt']).dt.days.head()
# or the time difference to today, or any other day of reference



(datetime.datetime.today() - data['issue_dt']).head()
(datetime.datetime.today() - data['issue_dt']).tail()