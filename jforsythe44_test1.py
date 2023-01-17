# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# You may want to define the dtypes for timeStamps

d=pd.read_csv("../input/trainView.csv",
  header=0,names=['train_id','status','next_station','service','dest','lon','lat','source',
  'track_change','track','date','timeStamp0','timeStamp1'],
  dtype={'train_id':str,'status':str,'next_station':str,'service':str,'dest':str,
  'lon':str,'lat':str,'source':str,'track_change':str,'track':str,
  'date':str,
  'timeStamp0':datetime.datetime,'timeStamp1':datetime.datetime})
#Only Norristown Train Line
nor = d[d.source == 'Norristown']
nor = pd.DataFrame(nor)
#Group Count by Status 
grouped = nor.groupby('status')
status_group = grouped.agg({'source': pd.Series.count})
status_group.rename(columns={'status': 'delay', 
                        'source': 'total_delay'}, inplace=True)
status_group.reset_index(inplace=True)
status_group['status'] = status_group.status.astype(int)
status_group = status_group.sort_values(by='status')
status_group.head()
import matplotlib.pyplot as plt
%matplotlib inline

status_chrt = status_group[status_group.status < 60]


with plt.style.context('fivethirtyeight'):
    plt.bar(status_chrt.status, status_chrt.total_delay, alpha=0.5)
    plt.ylabel('Count')
    plt.xlabel('Status')
#Percent of late (status>0) vs ontime (status=0)

#limit to 30 minutes for better chart view
status_chrt_30 = status_group[status_group.status < 30]
status = status_chrt_30.total_delay
total = status_chrt.total_delay.sum()

status_chrt_30['total_pct'] = status/total

with plt.style.context('fivethirtyeight'):
    plt.bar(status_chrt_30.status, status_chrt_30.total_pct, alpha=0.5)
    plt.title('Norristown Delay %age')
    plt.ylabel('delay_pct')
    plt.xlabel('Status')