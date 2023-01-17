# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#import dataset
df_callcenter = pd.read_csv ('../input/service-requests-received-by-the-oakland-call-center.csv')
df_callcenter.sample(3)
#check which values are missing 
df_null = df_callcenter.isnull()

#sum of the missing values
df_null.sum()
# first visualization: Histogram with count of council districts of the requests
plt.figure(figsize=(15,8))
first_diagram = sns.countplot(x ="COUNCILDISTRICT",  palette="rocket", data = df_callcenter).set_title('Count requests per Council District')

#second visualization: count of status
plt.figure(figsize=(15,8))
second_diagram = sns.countplot(y= 'STATUS',palette="rocket", data=df_callcenter).set_title('Count of the request status')
'''Fill in the Null / NaN values: DATETIMEINIT has no NaN values; 
DATETIMECLOSED is filled with the current business date
--> pd.to_datetime('now')
''' 

data = {'REQUESTID': df_callcenter['REQUESTID'],
        'REQCATEGORY': df_callcenter['REQCATEGORY'],
    'DATETIMEINIT': df_callcenter['DATETIMEINIT'],
 'DATETIMECLOSED': df_callcenter['DATETIMECLOSED'].fillna(pd.to_datetime('now'))}

df_callcenter_diff = pd.DataFrame(data, columns = ('REQUESTID','REQCATEGORY','DATETIMEINIT','DATETIMECLOSED'))
df_callcenter_diff.head(3)
# Test whether all NaN values were replaced
test_null = df_callcenter_diff.isnull()
test_null.sum()
from datetime import datetime
df_callcenter_diff['DATETIMEINIT_new'] = pd.to_datetime(df_callcenter_diff['DATETIMEINIT']).astype('datetime64[D]')
df_callcenter_diff['DATETIMECLOSED_new'] = pd.to_datetime(df_callcenter_diff['DATETIMECLOSED']).astype('datetime64[D]')
df_callcenter_diff['TIME_DIFF'] = df_callcenter_diff['DATETIMECLOSED_new'] - df_callcenter_diff['DATETIMEINIT_new']
df_callcenter_diff['TIME_DIFF_days'] = df_callcenter_diff['TIME_DIFF'].dt.days
df_callcenter_diff.sample(3)
help(plt.plot)
grouped = df_callcenter_diff.groupby('REQCATEGORY')
grouped_mean = grouped['TIME_DIFF_days'].agg(np.mean)

plt.figure(figsize=(15,8))
ax = grouped_mean.plot(kind = 'bar', color = 'r').set_title('Average time to close request per request category')
status_count = df_callcenter['STATUS'].value_counts()
#select distinct values of column 'STATUS'
test = df_callcenter['STATUS'].unique()
test
df = df_callcenter.groupby('STATUS')['REQUESTID'].nunique()
df
cd_val_count = df_callcenter['COUNCILDISTRICT'].value_counts()
cd_val_count
# Count of the values in attribute status
status_count = df_callcenter['STATUS'].value_counts()
status_count