# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#load in the data
inspections_dataset = pd.read_csv('../input/restaurant-scores-lives-standard.csv')

#need dates
inspections_dataset.inspection_date = pd.to_datetime(inspections_dataset.inspection_date)
#which restaurants have had the most severe violations
#really only care about routine inspections for this
inspections_dataset[(inspections_dataset.inspection_type=='Routine - Unscheduled') & (inspections_dataset.risk_category == 'High Risk')]\
.groupby(['business_id', 'business_name'])['inspection_id']\
.count().reset_index(name='no_severe_viol')\
.sort_values(ascending=False, by='no_severe_viol')\
.head(10)\
.plot.barh(x='business_name', y='no_severe_viol', legend=False, title='All-time Worst Offenders', color=['grey']).invert_yaxis()
#inspections in the last week and results
#not up to date, so what is the most recent inspection in the dataset?
most_recent_inspection = inspections_dataset['inspection_date'].max()
window_start = most_recent_inspection + dt.timedelta(days=-7)
time_window = pd.date_range(window_start, most_recent_inspection)

viol_by_cat = inspections_dataset[inspections_dataset.inspection_date.isin(time_window)]\
.groupby(['inspection_date', 'risk_category'])['business_id']\
.count().reset_index(name="count")

#we might not have inspections every day and risk category, need to add missing values

viol_by_date = pd.pivot_table(viol_by_cat, values='count', index='inspection_date', columns='risk_category')
viol_by_date = viol_by_date.reindex(time_window, fill_value = 0)
viol_by_date = viol_by_date[['Low Risk', 'Moderate Risk', 'High Risk']]

#eye searing, but effective...
viol_by_date.plot.bar(stacked=True, title='Most Recent Inspections', color=['green', 'yellow', 'red'])